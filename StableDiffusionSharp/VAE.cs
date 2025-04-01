using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace StableDiffusionSharp
{
	internal class VAE
	{
		private static GroupNorm Normalize(int in_channels, int num_groups = 32, float eps = 1e-6f, bool affine = true, Device? device = null, ScalarType? dtype = null)
		{
			return torch.nn.GroupNorm(num_groups: num_groups, num_channels: in_channels, eps: eps, affine: affine, device: device, dtype: dtype);
		}

		private class ResnetBlock : Module<Tensor, Tensor>
		{
			private readonly int in_channels;
			private readonly int out_channels;
			private readonly GroupNorm norm1;
			private readonly Conv2d conv1;
			private readonly GroupNorm norm2;
			private readonly Conv2d conv2;
			private readonly Module<Tensor, Tensor> nin_shortcut;
			private readonly SiLU swish;

			public ResnetBlock(int in_channels, int out_channels, Device? device = null, ScalarType? dtype = null) : base(nameof(AttnBlock))
			{
				this.in_channels = in_channels;
				this.out_channels = out_channels;
				this.norm1 = Normalize(in_channels, device: device, dtype: dtype);
				this.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size: 3, stride: 1, padding: 1, device: device, dtype: dtype);
				this.norm2 = Normalize(out_channels, device: device, dtype: dtype);
				this.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size: 3, stride: 1, padding: 1, device: device, dtype: dtype);

				if (this.in_channels != this.out_channels)
				{
					this.nin_shortcut = torch.nn.Conv2d(in_channels: in_channels, out_channels: out_channels, kernel_size: 1, device: device, dtype: dtype);
				}
				else
				{
					this.nin_shortcut = torch.nn.Identity();
				}

				this.swish = torch.nn.SiLU(inplace: true);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				Tensor hidden = x;
				hidden = this.norm1.forward(hidden);
				hidden = this.swish.forward(hidden);
				hidden = this.conv1.forward(hidden);
				hidden = this.norm2.forward(hidden);
				hidden = this.swish.forward(hidden);
				hidden = this.conv2.forward(hidden);
				if (this.in_channels != this.out_channels)
				{
					x = this.nin_shortcut.forward(x);
				}
				return x + hidden;
			}
		}

		private class AttnBlock : Module<Tensor, Tensor>
		{
			private readonly GroupNorm norm;
			private readonly Conv2d q;
			private readonly Conv2d k;
			private readonly Conv2d v;
			private readonly Conv2d proj_out;

			public AttnBlock(int in_channels, Device? device = null, ScalarType? dtype = null) : base(nameof(AttnBlock))
			{
				this.norm = Normalize(in_channels, device: device, dtype: dtype);
				this.q = Conv2d(in_channels, in_channels, kernel_size: 1, device: device, dtype: dtype);
				this.k = Conv2d(in_channels, in_channels, kernel_size: 1, device: device, dtype: dtype);
				this.v = Conv2d(in_channels, in_channels, kernel_size: 1, device: device, dtype: dtype);
				this.proj_out = Conv2d(in_channels, in_channels, kernel_size: 1, device: device, dtype: dtype);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using (NewDisposeScope())
				{
					var hidden = norm.forward(x);
					var q = this.q.forward(hidden);
					var k = this.k.forward(hidden);
					var v = this.v.forward(hidden);

					var (b, c, h, w) = (q.size(0), q.size(1), q.size(2), q.size(3));

					q = q.view(b, 1, h * w, c).contiguous();
					k = k.view(b, 1, h * w, c).contiguous();
					v = v.view(b, 1, h * w, c).contiguous();

					hidden = functional.scaled_dot_product_attention(q, k, v); // scale_factor is dim ** -0.5 per default

					hidden = hidden.view(b, c, h, w).contiguous();
					hidden = proj_out.forward(hidden);

					return (x + hidden).MoveToOuterDisposeScope();
				}
			}

		}

		private class Downsample : Module<Tensor, Tensor>
		{
			private readonly Conv2d? conv;
			private readonly bool with_conv;

			public Downsample(int in_channels, bool with_conv = true, Device? device = null, ScalarType? dtype = null) : base(nameof(Downsample))
			{
				this.with_conv = with_conv;
				if (with_conv)
				{
					this.conv = Conv2d(in_channels, in_channels, kernel_size: 3, stride: 2, device: device, dtype: dtype);

				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				if (this.with_conv && this.conv != null)
				{
					long[] pad = [0, 1, 0, 1];
					x = functional.pad(x, pad, mode: PaddingModes.Constant, value: 0);
					x = this.conv.forward(x);
				}
				else
				{
					x = torch.nn.functional.avg_pool2d(x, kernel_size: 2, stride: 2);
				}
				return x;
			}
		}

		private class Upsample : Module<Tensor, Tensor>
		{
			private readonly Conv2d? conv;
			private readonly bool with_conv;
			public Upsample(int in_channels, bool with_conv = true, Device? device = null, ScalarType? dtype = null) : base(nameof(Upsample))
			{
				this.with_conv = with_conv;
				if (with_conv)
				{
					this.conv = Conv2d(in_channels, in_channels, kernel_size: 3, padding: 1, device: device, dtype: dtype);
				}
				RegisterComponents();
			}
			public override Tensor forward(Tensor x)
			{
				var output = torch.nn.functional.interpolate(x, scale_factor: [2.0, 2.0], mode: InterpolationMode.Nearest);
				if (this.with_conv && this.conv != null)
				{
					output = this.conv.forward(output);
				}
				return output;
			}
		}

		private class VAEEncoder : Module<Tensor, Tensor>
		{
			private readonly int num_resolutions;
			private readonly int num_res_blocks;
			private readonly Conv2d conv_in;
			private readonly List<int> in_ch_mult;
			private readonly Sequential down;
			private readonly Sequential mid;
			private readonly GroupNorm norm_out;
			private readonly Conv2d conv_out;
			private readonly SiLU swish;
			private readonly int block_in;
			private readonly bool double_z;


			public VAEEncoder(int ch = 128, int[]? ch_mult = null, int num_res_blocks = 2, int in_channels = 3, int z_channels = 16, bool double_z = true, Device? device = null, ScalarType? dtype = null) : base(nameof(VAEEncoder))
			{
				this.double_z = double_z;
				ch_mult ??= [1, 2, 4, 4];
				this.num_resolutions = ch_mult.Length;
				this.num_res_blocks = num_res_blocks;

				// Input convolution
				conv_in = Conv2d(in_channels, ch, kernel_size: 3, stride: 1, padding: 1, device: device, dtype: dtype);

				// Downsampling layers
				in_ch_mult = [1, .. ch_mult];
				this.down = Sequential();

				block_in = ch * in_ch_mult[0];

				for (int i_level = 0; i_level < num_resolutions; i_level++)
				{
					var block = Sequential();
					var attn = Sequential();
					int block_out = ch * ch_mult[i_level];
					block_in = ch * in_ch_mult[i_level];
					for (int _ = 0; _ < num_res_blocks; _++)
					{
						block.append(new ResnetBlock(block_in, block_out, device: device, dtype: dtype));
						block_in = block_out;
					}

					var d = Sequential(
						("block", block),
						("attn", attn));

					if (i_level != num_resolutions - 1)
					{
						d.append("downsample", new Downsample(block_in, device: device, dtype: dtype));
					}
					this.down.append(d);
				}

				// Middle layers
				this.mid = Sequential(
					("block_1", new ResnetBlock(block_in, block_in, device: device, dtype: dtype)),
					("attn_1", new AttnBlock(block_in, device: device, dtype: dtype)),
					("block_2", new ResnetBlock(block_in, block_in, device: device, dtype: dtype)));


				// Output layers
				norm_out = Normalize(block_in, device: device, dtype: dtype);
				conv_out = Conv2d(block_in, (double_z ? 2 : 1) * z_channels, kernel_size: 3, stride: 1, padding: 1, device: device, dtype: dtype);
				swish = SiLU(inplace: true);

				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();

				// Downsampling
				var h = conv_in.forward(x);

				h = down.forward(h);

				// Middle layers
				h = mid.forward(h);

				// Output layers
				h = norm_out.forward(h);
				h = swish.forward(h);
				h = conv_out.forward(h);
				return h.MoveToOuterDisposeScope();
			}
		}

		private class VAEDecoder : Module<Tensor, Tensor>
		{
			private readonly int num_resolutions;
			private readonly int num_res_blocks;

			private readonly Conv2d conv_in;
			private readonly Sequential mid;

			private readonly Sequential up;

			private readonly GroupNorm norm_out;
			private readonly Conv2d conv_out;
			private readonly GELU swish;

			public VAEDecoder(int ch = 128, int out_ch = 3, int[]? ch_mult = null, int num_res_blocks = 2, int resolution = 256, int z_channels = 16, Device? device = null, ScalarType? dtype = null) : base(nameof(VAEDecoder))
			{
				ch_mult ??= [1, 2, 4, 4];
				this.num_resolutions = ch_mult.Length;
				this.num_res_blocks = num_res_blocks;
				int block_in = ch * ch_mult[this.num_resolutions - 1];

				int curr_res = resolution / (int)Math.Pow(2, num_resolutions - 1);
				// z to block_in
				this.conv_in = Conv2d(z_channels, block_in, kernel_size: 3, padding: 1, device: device, dtype: dtype);

				// middle
				this.mid = Sequential(
					("block_1", new ResnetBlock(block_in, block_in, device: device, dtype: dtype)),
					("attn_1", new AttnBlock(block_in, device: device, dtype: dtype)),
					("block_2", new ResnetBlock(block_in, block_in, device: device, dtype: dtype))
					);

				// upsampling
				this.up = Sequential();

				List<Sequential> list = new List<Sequential>();
				for (int i_level = this.num_resolutions - 1; i_level >= 0; i_level--)
				{
					var block = Sequential();

					int block_out = ch * ch_mult[i_level];

					for (int i_block = 0; i_block < num_res_blocks + 1; i_block++)
					{
						block.append(new ResnetBlock(block_in, block_out, device: device, dtype: dtype));
						block_in = block_out;
					}

					Sequential u = Sequential(("block", block));

					if (i_level != 0)
					{
						u.append("upsample", new Upsample(block_in, device: device, dtype: dtype));
						curr_res *= 2;
					}
					//this.up.append(u);
					list.Insert(0, u);
				}

				this.up = Sequential(list);

				// end
				this.norm_out = Normalize(block_in, device: device, dtype: dtype);
				this.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size: 3, stride: 1, padding: 1, device: device, dtype: dtype);
				this.swish = GELU(inplace: true);
				RegisterComponents();
			}

			public override Tensor forward(Tensor z)
			{
				// z to block_in
				Tensor hidden = this.conv_in.forward(z);

				// middle
				hidden = this.mid.forward(hidden);

				// upsampling
				foreach (Module<Tensor, Tensor> md in up.children().Reverse())
				{
					hidden = md.forward(hidden);
				}

				// end
				hidden = this.norm_out.forward(hidden);
				hidden = this.swish.forward(hidden);
				hidden = this.conv_out.forward(hidden);
				return hidden;
			}
		}

		internal class Decoder : Module<Tensor, Tensor>
		{
			private Sequential first_stage_model;

			public Decoder(int embed_dim = 4, int z_channels = 4, Device? device = null, ScalarType? dtype = null) : base(nameof(Decoder))
			{
				first_stage_model = Sequential(("post_quant_conv", Conv2d(embed_dim, z_channels, 1, device: device, dtype: dtype)), ("decoder", new VAEDecoder(z_channels: z_channels, device: device, dtype: dtype)));
				RegisterComponents();
			}

			public override Tensor forward(Tensor latents)
			{
				Device device = first_stage_model.parameters().First().device;
				ScalarType dtype = first_stage_model.parameters().First().dtype;
				latents = latents.to(dtype, device);
				return first_stage_model.forward(latents);
			}
		}

		internal class Encoder : Module<Tensor, Tensor>
		{
			private Sequential first_stage_model;
			public Encoder(int embed_dim = 4, int z_channels = 4, bool double_z = true, Device? device = null, ScalarType? dtype = null) : base(nameof(Encoder))
			{
				int factor = double_z ? 2 : 1;
				first_stage_model = Sequential(("encoder", new VAEEncoder(z_channels: z_channels, device: device, dtype: dtype)), ("quant_conv", Conv2d(factor * embed_dim, factor * z_channels, 1, device: device, dtype: dtype)));
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				Device device = first_stage_model.parameters().First().device;
				ScalarType dtype = first_stage_model.parameters().First().dtype;
				input = input.to(dtype, device);
				return first_stage_model.forward(input);
			}
		}

		private static long GetVideoCardMemory()
		{
			if (!torch.cuda.is_available())
			{
				return 0;
			}
			else
			{
				using (var factory = new SharpDX.DXGI.Factory1())
				{
					var adapter = factory.Adapters[0];
					using (var adapter3 = adapter.QueryInterface<SharpDX.DXGI.Adapter3>())
					{
						if (adapter3 == null)
						{
							throw new ArgumentException($"Adapter {adapter.Description.Description} not support");
						}
						var memoryInfo = adapter3.QueryVideoMemoryInfo(0, SharpDX.DXGI.MemorySegmentGroup.Local);
						long totalVRAM = adapter.Description.DedicatedVideoMemory;
						long usedVRAM = memoryInfo.CurrentUsage;
						long freeVRAM = memoryInfo.Budget - usedVRAM;
						return freeVRAM;
					}
				}
			}
		}

	}
}
