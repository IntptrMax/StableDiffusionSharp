using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace StableDiffusionSharp.Modules
{
	internal class VAE
	{
		private static GroupNorm Normalize(int in_channels, int num_groups = 32, float eps = 1e-6f, bool affine = true, Device? device = null, ScalarType? dtype = null)
		{
			return GroupNorm(num_groups: num_groups, num_channels: in_channels, eps: eps, affine: affine, device: device, dtype: dtype);
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
				norm1 = Normalize(in_channels, device: device, dtype: dtype);
				conv1 = Conv2d(in_channels, out_channels, kernel_size: 3, stride: 1, padding: 1, device: device, dtype: dtype);
				norm2 = Normalize(out_channels, device: device, dtype: dtype);
				conv2 = Conv2d(out_channels, out_channels, kernel_size: 3, stride: 1, padding: 1, device: device, dtype: dtype);

				if (this.in_channels != this.out_channels)
				{
					nin_shortcut = Conv2d(in_channels: in_channels, out_channels: out_channels, kernel_size: 1, device: device, dtype: dtype);
				}
				else
				{
					nin_shortcut = Identity();
				}

				swish = SiLU(inplace: true);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();
				Tensor hidden = x;
				hidden = norm1.forward(hidden);
				hidden = swish.forward(hidden);
				hidden = conv1.forward(hidden);
				hidden = norm2.forward(hidden);
				hidden = swish.forward(hidden);
				hidden = conv2.forward(hidden);
				if (in_channels != out_channels)
				{
					x = nin_shortcut.forward(x);
				}
				return (x + hidden).MoveToOuterDisposeScope();
			}

			protected override void Dispose(bool disposing)
			{
				if (disposing)
				{
					conv1?.Dispose();
					conv2?.Dispose();
					norm1?.Dispose();
					norm2?.Dispose();
					nin_shortcut?.Dispose();
					swish?.Dispose();
				}
				base.ClearModules();
				base.Dispose(disposing);
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
				norm = Normalize(in_channels, device: device, dtype: dtype);
				q = Conv2d(in_channels, in_channels, kernel_size: 1, device: device, dtype: dtype);
				k = Conv2d(in_channels, in_channels, kernel_size: 1, device: device, dtype: dtype);
				v = Conv2d(in_channels, in_channels, kernel_size: 1, device: device, dtype: dtype);
				proj_out = Conv2d(in_channels, in_channels, kernel_size: 1, device: device, dtype: dtype);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();

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

			protected override void Dispose(bool disposing)
			{
				if (disposing)
				{
					norm?.Dispose();
					q?.Dispose();
					k?.Dispose();
					v?.Dispose();
					proj_out?.Dispose();
					ClearModules();
				}
				base.Dispose(disposing);
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
					conv = Conv2d(in_channels, in_channels, kernel_size: 3, stride: 2, device: device, dtype: dtype);

				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				if (with_conv && conv != null)
				{
					long[] pad = new long[] { 0, 1, 0, 1 };
					x = functional.pad(x, pad, mode: PaddingModes.Constant, value: 0);
					x = conv.forward(x);
				}
				else
				{
					x = functional.avg_pool2d(x, kernel_size: 2, stride: 2);
				}
				return x;
			}

			protected override void Dispose(bool disposing)
			{
				if (disposing)
				{
					conv?.Dispose();
					ClearModules();
				}
				base.Dispose(disposing);
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
					conv = Conv2d(in_channels, in_channels, kernel_size: 3, padding: 1, device: device, dtype: dtype);
				}
				RegisterComponents();
			}
			public override Tensor forward(Tensor x)
			{
				Tensor output = functional.interpolate(x, scale_factor: new double[] { 2.0, 2.0 }, mode: InterpolationMode.Nearest);
				if (with_conv && conv != null)
				{
					output = conv.forward(output);
				}
				return output;
			}

			protected override void Dispose(bool disposing)
			{
				if (disposing)
				{
					conv?.Dispose();
					ClearModules();
				}
				base.Dispose(disposing);
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
				ch_mult ??= new int[] { 1, 2, 4, 4 };
				num_resolutions = ch_mult.Length;
				this.num_res_blocks = num_res_blocks;

				// Input convolution
				conv_in = Conv2d(in_channels, ch, kernel_size: 3, stride: 1, padding: 1, device: device, dtype: dtype);

				// Downsampling layers
				in_ch_mult = new List<int> { 1 };
				in_ch_mult.AddRange(ch_mult);
				down = Sequential();

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
					down.append(d);
				}

				// Middle layers
				mid = Sequential(
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
				// Downsampling
				x = conv_in.forward(x);
				x = down.forward(x);
				// Middle layers
				x = mid.forward(x);
				// Output layers
				x = norm_out.forward(x);
				x = swish.forward(x);
				x = conv_out.forward(x);
				return x.MoveToOuterDisposeScope();
			}

			protected override void Dispose(bool disposing)
			{
				if (disposing)
				{
					conv_in?.Dispose();
					down?.Dispose();
					mid?.Dispose();
					norm_out?.Dispose();
					conv_out?.Dispose();
					swish?.Dispose();
					ClearModules();
				}
				base.Dispose(disposing);
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
				ch_mult ??= new int[] { 1, 2, 4, 4 };
				num_resolutions = ch_mult.Length;
				this.num_res_blocks = num_res_blocks;
				int block_in = ch * ch_mult[num_resolutions - 1];

				int curr_res = resolution / (int)Math.Pow(2, num_resolutions - 1);
				// x to block_in
				conv_in = Conv2d(z_channels, block_in, kernel_size: 3, padding: 1, device: device, dtype: dtype);

				// middle
				mid = Sequential(
					("block_1", new ResnetBlock(block_in, block_in, device: device, dtype: dtype)),
					("attn_1", new AttnBlock(block_in, device: device, dtype: dtype)),
					("block_2", new ResnetBlock(block_in, block_in, device: device, dtype: dtype))
					);

				// upsampling
				up = Sequential();

				List<Sequential> list = new List<Sequential>();
				for (int i_level = num_resolutions - 1; i_level >= 0; i_level--)
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

				up = Sequential(list);

				// end
				norm_out = Normalize(block_in, device: device, dtype: dtype);
				conv_out = Conv2d(block_in, out_ch, kernel_size: 3, stride: 1, padding: 1, device: device, dtype: dtype);
				swish = GELU(inplace: true);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				// x to block_in
				x = conv_in.forward(x);

				// middle
				x = mid.forward(x);

				// upsampling
				foreach (Module<Tensor, Tensor> md in up.children().Reverse())
				{
					x = md.forward(x);
				}

				// end
				x = norm_out.forward(x);
				x = swish.forward(x);
				x = conv_out.forward(x);
				return x;
			}

			protected override void Dispose(bool disposing)
			{
				if (disposing)
				{
					conv_in?.Dispose();
					mid?.Dispose();
					up?.Dispose();
					norm_out?.Dispose();
					conv_out?.Dispose();
					swish?.Dispose();
					ClearModules();
				}
				base.Dispose(disposing);
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

			protected override void Dispose(bool disposing)
			{
				if (disposing)
				{
					first_stage_model?.Dispose();
					ClearModules();
				}
				base.Dispose(disposing);
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

			protected override void Dispose(bool disposing)
			{
				if (disposing)
				{
					first_stage_model?.Dispose();
					ClearModules();
				}
				base.Dispose(disposing);
			}
		}
	}
}
