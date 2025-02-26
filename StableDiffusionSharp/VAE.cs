using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace StableDiffusionSharp
{
	public class VAE
	{
		private static GroupNorm Normalize(int in_channels, int num_groups = 32)
		{
			return torch.nn.GroupNorm(num_groups: num_groups, num_channels: in_channels, eps: 1e-6, affine: true);
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

			public ResnetBlock(int in_channels, int out_channels) : base(nameof(AttnBlock))
			{
				this.in_channels = in_channels;
				this.out_channels = out_channels;
				this.norm1 = Normalize(in_channels);
				this.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size: 3, stride: 1, padding: 1);
				this.norm2 = Normalize(out_channels);
				this.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size: 3, stride: 1, padding: 1);

				if (this.in_channels != this.out_channels)
				{
					this.nin_shortcut = torch.nn.Conv2d(in_channels: in_channels, out_channels: out_channels, kernel_size: 1);
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

			public AttnBlock(int in_channels) : base(nameof(AttnBlock))
			{
				this.norm = Normalize(in_channels);
				this.q = Conv2d(in_channels, in_channels, kernel_size: 1);
				this.k = Conv2d(in_channels, in_channels, kernel_size: 1);
				this.v = Conv2d(in_channels, in_channels, kernel_size: 1);
				this.proj_out = Conv2d(in_channels, in_channels, kernel_size: 1);
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

					hidden = functional.scaled_dot_product_attention(q, k, v); // scale is dim ** -0.5 per default

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

			public Downsample(int in_channels, bool with_conv = true) : base(nameof(Downsample))
			{
				this.with_conv = with_conv;
				if (with_conv)
				{
					this.conv = Conv2d(in_channels, in_channels, kernel_size: 3, stride: 2);

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
			public Upsample(int in_channels, bool with_conv = true) : base(nameof(Upsample))
			{
				this.with_conv = with_conv;
				if (with_conv)
				{
					this.conv = Conv2d(in_channels, in_channels, kernel_size: 3, padding: 1);
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

			public VAEEncoder(int ch = 128, int[]? ch_mult = null, int num_res_blocks = 2, int in_channels = 3, int z_channels = 16) : base(nameof(VAEEncoder))
			{
				ch_mult ??= [1, 2, 4, 4];
				this.num_resolutions = ch_mult.Length;
				this.num_res_blocks = num_res_blocks;

				// Input convolution
				conv_in = Conv2d(in_channels, ch, kernel_size: 3, stride: 1, padding: 1);

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
						block.append(new ResnetBlock(block_in, block_out));
						block_in = block_out;
					}

					var d = Sequential(
						("block", block),
						("attn", attn));

					if (i_level != num_resolutions - 1)
					{
						d.append("downsample", new Downsample(block_in));
					}
					this.down.append(d);
				}

				// Middle layers
				this.mid = Sequential(
					("block_1", new ResnetBlock(block_in, block_in)),
					("attn_1", new AttnBlock(block_in)),
					("block_2", new ResnetBlock(block_in, block_in)));


				// Output layers
				norm_out = Normalize(block_in);
				conv_out = Conv2d(block_in, 2 * z_channels, kernel_size: 3, stride: 1, padding: 1);
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

			public VAEDecoder(int ch = 128, int out_ch = 3, int[]? ch_mult = null, int num_res_blocks = 2, int resolution = 256, int z_channels = 16) : base(nameof(VAEDecoder))
			{
				ch_mult ??= [1, 2, 4, 4];
				this.num_resolutions = ch_mult.Length;
				this.num_res_blocks = num_res_blocks;
				int block_in = ch * ch_mult[this.num_resolutions - 1];

				int curr_res = resolution / (int)Math.Pow(2, num_resolutions - 1);
				// z to block_in
				this.conv_in = Conv2d(z_channels, block_in, kernel_size: 3, padding: 1);

				// middle
				this.mid = Sequential(
					("block_1", new ResnetBlock(block_in, block_in)),
					("attn_1", new AttnBlock(block_in)),
					("block_2", new ResnetBlock(block_in, block_in))
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
						block.append(new ResnetBlock(block_in, block_out));
						block_in = block_out;
					}

					Sequential u = Sequential(("block", block));

					if (i_level != 0)
					{
						u.append("upsample", new Upsample(block_in));
						curr_res *= 2;
					}
					//this.up.append(u);
					list.Insert(0, u);
				}

				this.up = Sequential(list);

				// end
				this.norm_out = Normalize(block_in);
				this.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size: 3, stride: 1, padding: 1);
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

		public class Decoder : Module<Tensor, Tensor>
		{
			private Sequential first_stage_model;
			public Decoder(int z_channels = 4) : base(nameof(Decoder))
			{
				first_stage_model = Sequential(("post_quant_conv", Conv2d(z_channels, z_channels, 1)), ("decoder", new VAEDecoder(z_channels: z_channels)));
				RegisterComponents();
			}

			public override Tensor forward(Tensor latents)
			{
				Tensor output = latents / 0.18215f;
				return first_stage_model.forward(output);
			}
		}

		public class Encoder : Module<Tensor, Tensor>
		{
			private Sequential first_stage_model;
			public Encoder() : base(nameof(Encoder))
			{
				first_stage_model = Sequential(("encoder", new VAEEncoder()));
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				return first_stage_model.forward(input);
			}
		}

	}
}
