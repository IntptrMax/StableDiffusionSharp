using System;
using TorchSharp;
using TorchSharp.Modules;
using static Tensorboard.CostGraphDef.Types;
using static Tensorboard.TensorShapeProto.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace StableDiffusionSharp.Modules
{
	internal class CrossAttention : Module<Tensor, Tensor, Tensor>
	{
		private readonly Linear to_q;
		private readonly Linear to_k;
		private readonly Linear to_v;
		private readonly Sequential to_out;
		private readonly long n_heads_;
		private readonly long d_head;
		private readonly bool causal_mask_;

		public CrossAttention(long channels, long d_cross, long n_heads, bool causal_mask = false, bool in_proj_bias = false, bool out_proj_bias = true, float dropout_p = 0.0f, Device? device = null, ScalarType? dtype = null) : base(nameof(CrossAttention))
		{
			to_q = Linear(channels, channels, hasBias: in_proj_bias, device: device, dtype: dtype);
			to_k = Linear(d_cross, channels, hasBias: in_proj_bias, device: device, dtype: dtype);
			to_v = Linear(d_cross, channels, hasBias: in_proj_bias, device: device, dtype: dtype);
			to_out = Sequential(Linear(channels, channels, hasBias: out_proj_bias, device: device, dtype: dtype), Dropout(dropout_p, inplace: false));
			n_heads_ = n_heads;
			d_head = channels / n_heads;
			causal_mask_ = causal_mask;
			RegisterComponents();
		}

		public override Tensor forward(Tensor x, Tensor y)
		{
			using (NewDisposeScope())
			{
				long[] input_shape = x.shape;
				long batch_size = input_shape[0];
				long sequence_length = input_shape[1];

				long[] interim_shape = new long[] { batch_size, -1, n_heads_, d_head };
				Tensor q = to_q.forward(x);
				Tensor k = to_k.forward(y);
				Tensor v = to_v.forward(y);

				q = q.view(interim_shape).transpose(1, 2);
				k = k.view(interim_shape).transpose(1, 2);
				v = v.view(interim_shape).transpose(1, 2);
				Tensor output = functional.scaled_dot_product_attention(q, k, v, is_casual: causal_mask_);
				output = output.transpose(1, 2).reshape(input_shape);
				output = to_out.forward(output);
				return output.MoveToOuterDisposeScope();
			}
		}
	}

	internal class ResnetBlock : Module<Tensor, Tensor, Tensor>
	{
		private readonly int in_channels;
		private readonly int out_channels;

		private readonly Module<Tensor, Tensor> skip_connection;
		private readonly Sequential emb_layers;
		private readonly Sequential in_layers;
		private readonly Sequential out_layers;

		public ResnetBlock(int in_channels, int out_channels, double dropout = 0.0, int temb_channels = 1280, Device? device = null, ScalarType? dtype = null) : base(nameof(ResnetBlock))
		{
			this.in_channels = in_channels;
			out_channels = out_channels < 1 ? in_channels : out_channels;
			this.out_channels = out_channels;

			in_layers = Sequential(GroupNorm(32, in_channels, device: device, dtype: dtype), SiLU(), Conv2d(in_channels, out_channels, kernel_size: 3, stride: 1, padding: 1, device: device, dtype: dtype));

			if (temb_channels > 0)
			{
				emb_layers = Sequential(SiLU(), Linear(temb_channels, out_channels, device: device, dtype: dtype));
			}

			out_layers = Sequential(GroupNorm(32, out_channels, device: device, dtype: dtype), SiLU(), Dropout(dropout), Conv2d(out_channels, out_channels, kernel_size: 3, stride: 1, padding: 1, device: device, dtype: dtype));

			if (this.in_channels != this.out_channels)
			{
				skip_connection = Conv2d(in_channels: in_channels, out_channels: this.out_channels, kernel_size: 1, stride: 1, device: device, dtype: dtype);
			}
			else
			{
				skip_connection = Identity();
			}

			RegisterComponents();
		}

		public override Tensor forward(Tensor x, Tensor time)
		{
			using (NewDisposeScope())
			{
				Tensor hidden = x;
				hidden = in_layers.forward(hidden);

				if (time is not null)
				{
					time = emb_layers.forward(time);
					hidden = hidden + time.unsqueeze(-1).unsqueeze(-1);
				}

				hidden = out_layers.forward(hidden);
				if (in_channels != out_channels)
				{
					x = skip_connection.forward(x);
				}
				return (x + hidden).MoveToOuterDisposeScope();
			}
		}
	}

	internal class TransformerBlock : Module<Tensor, Tensor, Tensor>
	{
		private LayerNorm norm1;
		private CrossAttention attn1;
		private LayerNorm norm2;
		private CrossAttention attn2;
		private LayerNorm norm3;
		private FeedForward ff;

		public TransformerBlock(int channels, int n_cross, int n_head, Device? device = null, ScalarType? dtype = null) : base(nameof(TransformerBlock))
		{
			norm1 = LayerNorm(channels, device: device, dtype: dtype);
			attn1 = new CrossAttention(channels, channels, n_head, device: device, dtype: dtype);
			norm2 = LayerNorm(channels, device: device, dtype: dtype);
			attn2 = new CrossAttention(channels, n_cross, n_head, device: device, dtype: dtype);
			norm3 = LayerNorm(channels, device: device, dtype: dtype);
			ff = new FeedForward(channels, glu: true, device: device, dtype: dtype);
			RegisterComponents();
		}
		public override Tensor forward(Tensor x, Tensor context)
		{
			var residue_short = x;
			x = norm1.forward(x);
			x = attn1.forward(x, x);
			x += residue_short;
			residue_short = x;
			x = norm2.forward(x);
			x = attn2.forward(x, context);
			x += residue_short;
			residue_short = x;
			x = norm3.forward(x);
			x = ff.forward(x);
			x += residue_short;
			return x.MoveToOuterDisposeScope();
		}
	}

	internal class SpatialTransformer : Module<Tensor, Tensor, Tensor>
	{
		private readonly GroupNorm norm;
		private readonly Module<Tensor, Tensor> proj_in;
		private readonly Module<Tensor, Tensor> proj_out;
		private readonly ModuleList<Module<Tensor, Tensor, Tensor>> transformer_blocks;
		private readonly bool use_linear;

		public SpatialTransformer(int channels, int n_cross, int n_head, int num_atten_blocks, float drop_out = 0.0f, bool use_linear = false, Device? device = null, ScalarType? dtype = null) : base(nameof(SpatialTransformer))
		{
			norm = Normalize(channels, device: device, dtype: dtype);
			this.use_linear = use_linear;
			proj_in = use_linear ? Linear(channels, channels, device: device, dtype: dtype) : Conv2d(channels, channels, kernel_size: 1, device: device, dtype: dtype);
			proj_out = use_linear ? Linear(channels, channels, device: device, dtype: dtype) : Conv2d(channels, channels, kernel_size: 1, device: device, dtype: dtype);
			transformer_blocks = new ModuleList<Module<Tensor, Tensor, Tensor>>();
			for (int i = 0; i < num_atten_blocks; i++)
			{
				transformer_blocks.Add(new TransformerBlock(channels, n_cross, n_head, device: device, dtype: dtype));
			}
			RegisterComponents();
		}

		public override Tensor forward(Tensor x, Tensor context)
		{
			using (NewDisposeScope())
			{
				long n = x.shape[0];
				long c = x.shape[1];
				long h = x.shape[2];
				long w = x.shape[3];

				Tensor residue_short = x;
				x = norm.forward(x);

				if (!use_linear)
				{
					x = proj_in.forward(x);
				}

				x = x.view(new long[] { n, c, h * w });
				x = x.transpose(-1, -2);

				if (use_linear)
				{
					x = proj_in.forward(x);
				}

				foreach (Module<Tensor, Tensor, Tensor> layer in transformer_blocks)
				{
					x = layer.forward(x, context);
				}

				if (use_linear)
				{
					x = proj_out.forward(x);
				}
				x = x.transpose(-1, -2);
				x = x.view(new long[] { n, c, h, w });
				if (!use_linear)
				{
					x = proj_out.forward(x);
				}

				residue_short = residue_short + x;
				return residue_short.MoveToOuterDisposeScope();
			}
		}

		private static GroupNorm Normalize(int in_channels, int num_groups = 32, float eps = 1e-6f, bool affine = true, Device? device = null, ScalarType? dtype = null)
		{
			return GroupNorm(num_groups: 32, num_channels: in_channels, eps: eps, affine: affine, device: device, dtype: dtype);
		}

	}

	internal class Upsample : Module<Tensor, Tensor>
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
			var output = functional.interpolate(x, scale_factor: new double[] { 2.0, 2.0 }, mode: InterpolationMode.Nearest);
			if (with_conv && conv is not null)
			{
				output = conv.forward(output);
			}
			return output;
		}
	}

	internal class Downsample : Module<Tensor, Tensor>
	{
		private readonly Conv2d op;
		public Downsample(int in_channels, Device? device = null, ScalarType? dtype = null) : base(nameof(Downsample))
		{
			op = Conv2d(in_channels: in_channels, out_channels: in_channels, kernel_size: 3, stride: 2, padding: 1, device: device, dtype: dtype);
			RegisterComponents();
		}
		public override Tensor forward(Tensor x)
		{
			x = op.forward(x);
			return x;
		}
	}

	internal class TimestepEmbedSequential : Sequential<Tensor, Tensor, Tensor, Tensor>
	{
		internal TimestepEmbedSequential(params (string name, Module)[] modules) : base(modules)
		{
			RegisterComponents();
		}

		internal TimestepEmbedSequential(params Module[] modules) : base(modules)
		{
			RegisterComponents();
		}

		public override Tensor forward(Tensor x, Tensor context, Tensor time)
		{
			using (NewDisposeScope())
			{
				foreach (var layer in children())
				{
					switch (layer)
					{
						case ResnetBlock res:
							x = res.call(x, time);
							break;
						case SpatialTransformer abl:
							x = abl.call(x, context);
							break;
						case Module<Tensor, Tensor> m:
							x = m.call(x);
							break;
					}
				}
				return x.MoveToOuterDisposeScope();
			}
		}
	}

	internal class GEGLU : Module<Tensor, Tensor>
	{
		private readonly Linear proj;
		public GEGLU(int dim_in, int dim_out, Device? device = null, ScalarType? dtype = null) : base(nameof(GEGLU))
		{
			proj = Linear(dim_in, dim_out * 2, device: device, dtype: dtype);
			RegisterComponents();
		}

		public override Tensor forward(Tensor x)
		{
			using (NewDisposeScope())
			{
				Tensor[] result = proj.forward(x).chunk(2, dim: -1);
				x = result[0];
				Tensor gate = result[1];
				return (x * functional.gelu(gate)).MoveToOuterDisposeScope();
			}
		}
	}

	internal class FeedForward : Module<Tensor, Tensor>
	{
		private readonly Sequential net;

		public FeedForward(int dim, int? dim_out = null, int mult = 4, bool glu = true, float dropout = 0.0f, Device? device = null, ScalarType? dtype = null) : base(nameof(FeedForward))
		{
			int inner_dim = dim * mult;
			int dim_ot = dim_out ?? dim;
			Module<Tensor, Tensor> project_in = glu ? new GEGLU(dim, inner_dim, device: device, dtype: dtype) : Sequential(nn.Linear(dim, inner_dim, device: device, dtype: dtype), nn.GELU());
			net = Sequential(project_in, Dropout(dropout), Linear(inner_dim, dim_ot, device: device, dtype: dtype));
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			return net.forward(input);
		}
	}

	internal class SDUnet : Module<Tensor, Tensor, Tensor, Tensor, Tensor>
	{
		private class UNet : Module<Tensor, Tensor, Tensor, Tensor>
		{
			private readonly int ch;
			private readonly int time_embed_dim;
			private readonly int in_channels;
			private readonly bool use_timestep;

			private readonly Sequential time_embed;
			private readonly ModuleList<TimestepEmbedSequential> input_blocks;
			private readonly TimestepEmbedSequential middle_block;
			private readonly ModuleList<TimestepEmbedSequential> output_blocks;
			private readonly Sequential @out;

			public UNet(int model_channels, int in_channels, int[]? channel_mult = null, int num_res_blocks = 2, int num_atten_blocks = 1, int context_dim = 768, int num_heads = 8, float dropout = 0.0f, bool use_timestep = true, Device? device = null, ScalarType? dtype = null) : base(nameof(UNet))
			{
				bool mask = false;
				channel_mult = channel_mult ?? new int[] { 1, 2, 4, 4 };

				ch = model_channels;
				time_embed_dim = model_channels * 4;
				this.in_channels = in_channels;
				this.use_timestep = use_timestep;

				List<int> input_block_channels = new List<int> { model_channels };

				if (use_timestep)
				{
					// timestep embedding
					time_embed = Sequential(new Module<Tensor, Tensor>[] { Linear(model_channels, time_embed_dim, device: device, dtype: dtype), SiLU(), Linear(time_embed_dim, time_embed_dim, device: device, dtype: dtype) });
				}

				// downsampling
				input_blocks = new ModuleList<TimestepEmbedSequential>();
				input_blocks.Add(new TimestepEmbedSequential(Conv2d(in_channels, ch, kernel_size: 3, padding: 1, device: device, dtype: dtype)));

				for (int i = 0; i < channel_mult.Length; i++)
				{
					int in_ch = model_channels * channel_mult[i > 0 ? i - 1 : i];
					int out_ch = model_channels * channel_mult[i];

					for (int j = 0; j < num_res_blocks; j++)
					{
						input_blocks.Add(new TimestepEmbedSequential(new ResnetBlock(in_ch, out_ch, dropout, time_embed_dim, device: device, dtype: dtype), i < channel_mult.Length - 1 ? new SpatialTransformer(out_ch, context_dim, num_heads, num_atten_blocks, dropout, device: device, dtype: dtype) : Identity()));
						input_block_channels.Add(in_ch);
						in_ch = out_ch;
					}
					if (i < channel_mult.Length - 1)
					{
						input_blocks.Add(new TimestepEmbedSequential(Sequential(("op", Conv2d(out_ch, out_ch, 3, stride: 2, padding: 1, device: device, dtype: dtype)))));
						input_block_channels.Add(out_ch);
					}
				}

				// middle block
				middle_block = new TimestepEmbedSequential(new ResnetBlock(time_embed_dim, time_embed_dim, dropout, time_embed_dim, device: device, dtype: dtype), new SpatialTransformer(time_embed_dim, context_dim, num_heads, num_atten_blocks, dropout, device: device, dtype: dtype), new ResnetBlock(1280, 1280, device: device, dtype: dtype));

				// upsampling
				var reversed_mult = channel_mult.Reverse().ToList();
				int prev_channels = time_embed_dim;
				output_blocks = new ModuleList<TimestepEmbedSequential>();
				for (int i = 0; i < reversed_mult.Count; i++)
				{
					int mult = reversed_mult[i];
					int current_channels = model_channels * mult;
					int down_stage_index = channel_mult.Length - 1 - i;
					int skip_channels = model_channels * channel_mult[down_stage_index];
					bool has_atten = i >= 1;

					for (int j = 0; j < num_res_blocks + 1; j++)
					{
						int current_skip = skip_channels;
						if (j == num_res_blocks && i < reversed_mult.Count - 1)
						{
							int next_down_stage_index = channel_mult.Length - 1 - (i + 1);
							current_skip = model_channels * channel_mult[next_down_stage_index];
						}

						int input_channels = prev_channels + current_skip;
						bool has_upsample = j == num_res_blocks && i != reversed_mult.Count - 1;

						if (has_atten)
						{
							output_blocks.Add(new TimestepEmbedSequential(
								new ResnetBlock(input_channels, current_channels, dropout, time_embed_dim, device: device, dtype: dtype),
								new SpatialTransformer(current_channels, context_dim, num_heads, num_atten_blocks, dropout, device: device, dtype: dtype),
								has_upsample ? new Upsample(current_channels, device: device, dtype: dtype) : Identity()));
						}
						else
						{
							output_blocks.Add(new TimestepEmbedSequential(
								new ResnetBlock(input_channels, current_channels, dropout, time_embed_dim, device: device, dtype: dtype),
								has_upsample ? new Upsample(current_channels, device: device, dtype: dtype) : Identity()));
						}

						prev_channels = current_channels;
					}
				}

				@out = Sequential(GroupNorm(32, model_channels, device: device, dtype: dtype), SiLU(), Conv2d(model_channels, in_channels, kernel_size: 3, padding: 1, device: device, dtype: dtype));

				RegisterComponents();

			}
			public override Tensor forward(Tensor x, Tensor context, Tensor time)
			{
				using (NewDisposeScope())
				{
					time = time_embed.forward(time);

					List<Tensor> skip_connections = new List<Tensor>();
					foreach (TimestepEmbedSequential layers in input_blocks)
					{
						x = layers.forward(x, context, time);
						skip_connections.Add(x);
					}
					x = middle_block.forward(x, context, time);
					foreach (TimestepEmbedSequential layers in output_blocks)
					{
						Tensor index = skip_connections.Last();
						x = cat(new Tensor[] { x, index }, 1);
						skip_connections.RemoveAt(skip_connections.Count - 1);
						x = layers.forward(x, context, time);
					}

					x = @out.forward(x);
					return x.MoveToOuterDisposeScope();
				}
			}
		}

		private class Model : Module<Tensor, Tensor, Tensor, Tensor>
		{
			private readonly UNet diffusion_model;

			public Model(int model_channels, int in_channels, int num_heads = 8, int context_dim = 768, float dropout = 0.0f, bool use_timestep = true, Device? device = null, ScalarType? dtype = null) : base(nameof(SDUnet))
			{
				diffusion_model = new UNet(model_channels, in_channels, context_dim: context_dim, num_heads: num_heads, dropout: dropout, use_timestep: use_timestep, device: device, dtype: dtype);
				RegisterComponents();
			}

			public override Tensor forward(Tensor latent, Tensor context, Tensor time)
			{
				return diffusion_model.forward(latent, context, time);
			}
		}

		private readonly Model model;

		public SDUnet(int model_channels, int in_channels, int num_heads = 8, int context_dim = 768, float dropout = 0.0f, bool use_timestep = true, Device? device = null, ScalarType? dtype = null) : base(nameof(SDUnet))
		{
			model = new Model(model_channels, in_channels, context_dim: context_dim, num_heads: num_heads, dropout: dropout, use_timestep: use_timestep, device: device, dtype: dtype);
			RegisterComponents();
		}

		public override Tensor forward(Tensor latent, Tensor context, Tensor time, Tensor y)
		{
			Device device = model.parameters().First().device;
			ScalarType dtype = model.parameters().First().dtype;

			latent = latent.to(dtype, device);
			time = time.to(dtype, device);
			context = context.to(dtype, device);
			return model.forward(latent, context, time);
		}
	}

	internal class SDXLUnet : Module<Tensor, Tensor, Tensor, Tensor, Tensor>
	{
		private class UNet : Module<Tensor, Tensor, Tensor, Tensor, Tensor>
		{
			private readonly int ch;
			private readonly int time_embed_dim;
			private readonly int in_channels;
			private readonly bool use_timestep;

			private readonly Sequential time_embed;
			private readonly Sequential label_emb;
			private readonly ModuleList<TimestepEmbedSequential> input_blocks;
			private readonly TimestepEmbedSequential middle_block;
			private readonly ModuleList<TimestepEmbedSequential> output_blocks;
			private readonly Sequential @out;


			public UNet(int model_channels, int in_channels, int[]? channel_mult = null, int num_res_blocks = 2, int context_dim = 768, int adm_in_channels = 2816, int num_heads = 20, float dropout = 0.0f, bool use_timestep = true, Device? device = null, ScalarType? dtype = null) : base(nameof(SDUnet))
			{
				channel_mult = channel_mult ?? new int[] { 1, 2, 4 };

				ch = model_channels;
				time_embed_dim = model_channels * 4;
				this.in_channels = in_channels;
				this.use_timestep = use_timestep;

				bool useLinear = true;
				bool mask = false;

				List<int> input_block_channels = new List<int> { model_channels };

				if (use_timestep)
				{
					int time_embed_dim = model_channels * 4;
					time_embed = Sequential(Linear(model_channels, time_embed_dim, device: device, dtype: dtype), SiLU(), Linear(time_embed_dim, time_embed_dim, device: device, dtype: dtype));
					label_emb = Sequential(Sequential(Linear(adm_in_channels, time_embed_dim, device: device, dtype: dtype), SiLU(), Linear(time_embed_dim, time_embed_dim, device: device, dtype: dtype)));
				}

				// downsampling
				input_blocks = new ModuleList<TimestepEmbedSequential>();
				input_blocks.Add(new TimestepEmbedSequential(Conv2d(in_channels, ch, kernel_size: 3, padding: 1, device: device, dtype: dtype)));

				input_blocks.Add(new TimestepEmbedSequential(new ResnetBlock(320, 320, device: device, dtype: dtype)));
				input_blocks.Add(new TimestepEmbedSequential(new ResnetBlock(320, 320, device: device, dtype: dtype)));
				input_blocks.Add(new TimestepEmbedSequential(new Downsample(320, device: device, dtype: dtype)));

				input_blocks.Add(new TimestepEmbedSequential(new ResnetBlock(320, 640, device: device, dtype: dtype), new SpatialTransformer(640, 2048, num_heads, 2, 0, useLinear, device: device, dtype: dtype)));
				input_blocks.Add(new TimestepEmbedSequential(new ResnetBlock(640, 640, device: device, dtype: dtype), new SpatialTransformer(640, 2048, num_heads, 2, 0, useLinear, device: device, dtype: dtype)));
				input_blocks.Add(new TimestepEmbedSequential(new Downsample(640, device: device, dtype: dtype)));

				input_blocks.Add(new TimestepEmbedSequential(new ResnetBlock(640, 1280, device: device, dtype: dtype), new SpatialTransformer(1280, 2048, num_heads, 10, 0, useLinear, device: device, dtype: dtype)));
				input_blocks.Add(new TimestepEmbedSequential(new ResnetBlock(1280, 1280, device: device, dtype: dtype), new SpatialTransformer(1280, 2048, num_heads, 10, 0, useLinear, device: device, dtype: dtype)));

				// mid_block
				middle_block = new TimestepEmbedSequential(new ResnetBlock(1280, 1280, device: device, dtype: dtype), new SpatialTransformer(1280, 2048, num_heads, 10, 0, useLinear, device: device, dtype: dtype), new ResnetBlock(1280, 1280, device: device, dtype: dtype));

				// upsampling
				output_blocks = new ModuleList<TimestepEmbedSequential>();
				output_blocks.Add(new TimestepEmbedSequential(new ResnetBlock(2560, 1280, device: device, dtype: dtype), new SpatialTransformer(1280, 2048, num_heads, 10, 0, useLinear, device: device, dtype: dtype)));
				output_blocks.Add(new TimestepEmbedSequential(new ResnetBlock(2560, 1280, device: device, dtype: dtype), new SpatialTransformer(1280, 2048, num_heads, 10, 0, useLinear, device: device, dtype: dtype)));
				output_blocks.Add(new TimestepEmbedSequential(new ResnetBlock(1920, 1280, device: device, dtype: dtype), new SpatialTransformer(1280, 2048, num_heads, 10, 0, useLinear, device: device, dtype: dtype), new Upsample(1280, device: device, dtype: dtype)));

				output_blocks.Add(new TimestepEmbedSequential(new ResnetBlock(1920, 640, device: device, dtype: dtype), new SpatialTransformer(640, 2048, num_heads, 2, 0, useLinear, device: device, dtype: dtype)));
				output_blocks.Add(new TimestepEmbedSequential(new ResnetBlock(1280, 640, device: device, dtype: dtype), new SpatialTransformer(640, 2048, num_heads, 2, 0, useLinear, device: device, dtype: dtype)));
				output_blocks.Add(new TimestepEmbedSequential(new ResnetBlock(960, 640, device: device, dtype: dtype), new SpatialTransformer(640, 2048, num_heads, 2, 0, useLinear, device: device, dtype: dtype), new Upsample(640, device: device, dtype: dtype)));

				output_blocks.Add(new TimestepEmbedSequential(new ResnetBlock(960, 320, device: device, dtype: dtype)));
				output_blocks.Add(new TimestepEmbedSequential(new ResnetBlock(640, 320, device: device, dtype: dtype)));
				output_blocks.Add(new TimestepEmbedSequential(new ResnetBlock(640, 320, device: device, dtype: dtype)));

				@out = Sequential(GroupNorm(32, model_channels, device: device, dtype: dtype), SiLU(), Conv2d(model_channels, in_channels, kernel_size: 3, padding: 1, device: device, dtype: dtype));

				RegisterComponents();
			}

			public override Tensor forward(Tensor x, Tensor context, Tensor time, Tensor y)
			{
				using (NewDisposeScope())
				{
					int dim = 512;
					Tensor embed = time_embed.forward(time);
					Tensor time_ids = tensor(new float[] { dim, dim, 0, 0, dim, dim }, embed.dtype, embed.device).repeat(new long[] { 2, 1 });
					Tensor time_embeds = get_timestep_embedding(time_ids.flatten(), dim / 2, true, 0, 1);
					time_embeds = time_embeds.reshape(new long[] { 2, -1 });
					y = cat(new Tensor[] { y, time_embeds }, dim: -1);
					Tensor label_embed = label_emb.forward(y.to(embed.dtype, embed.device));
					embed = embed + label_embed;

					List<Tensor> skip_connections = new List<Tensor>();
					foreach (TimestepEmbedSequential layers in input_blocks)
					{
						x = layers.forward(x, context, embed);
						skip_connections.Add(x);
					}
					x = middle_block.forward(x, context, embed);
					foreach (TimestepEmbedSequential layers in output_blocks)
					{
						Tensor index = skip_connections.Last();
						x = cat(new Tensor[] { x, index }, 1);
						skip_connections.RemoveAt(skip_connections.Count - 1);
						x = layers.forward(x, context, embed);
					}

					x = @out.forward(x);
					return x.MoveToOuterDisposeScope();
				}

			}
		}

		private class Model : Module<Tensor, Tensor, Tensor, Tensor, Tensor>
		{
			private UNet diffusion_model;
			public Model(int model_channels, int in_channels, int num_heads = 20, int context_dim = 2048, int adm_in_channels = 2816, float dropout = 0.0f, bool use_timestep = true, Device? device = null, ScalarType? dtype = null) : base(nameof(SDUnet))
			{
				diffusion_model = new UNet(model_channels, in_channels, context_dim: context_dim, adm_in_channels: adm_in_channels, num_heads: num_heads, dropout: dropout, use_timestep: use_timestep, device: device, dtype: dtype);
				RegisterComponents();
			}

			public override Tensor forward(Tensor latent, Tensor context, Tensor time, Tensor y)
			{
				latent = diffusion_model.forward(latent, context, time, y);
				return latent;
			}
		}

		private readonly Model model;

		public SDXLUnet(int model_channels, int in_channels, int num_heads = 20, int context_dim = 2048, int adm_in_channels = 2816, float dropout = 0.0f, bool use_timestep = true, Device? device = null, ScalarType? dtype = null) : base(nameof(SDUnet))
		{
			model = new Model(model_channels, in_channels, context_dim: context_dim, adm_in_channels: adm_in_channels, num_heads: num_heads, dropout: dropout, use_timestep: use_timestep, device: device, dtype: dtype);
			RegisterComponents();
		}

		public override Tensor forward(Tensor latent, Tensor context, Tensor time, Tensor y)
		{
			Device device = model.parameters().First().device;
			ScalarType dtype = model.parameters().First().dtype;

			latent = latent.to(dtype, device);
			time = time.to(dtype, device);
			y = y.to(dtype, device);
			context = context.to(dtype, device);

			latent = model.forward(latent, context, time, y);
			return latent;
		}

		/// <summary>
		/// This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
		/// </summary>
		/// <param name="timesteps">a 1-D Tensor of N indices, one per batch element. These may be fractional.</param>
		/// <param name="embedding_dim">the dimension of the output.</param>
		/// <param name="flip_sin_to_cos">Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)</param>
		/// <param name="downscale_freq_shiftt">Controls the delta between frequencies between dimensions</param>
		/// <param name="scale">Scaling factor applied to the embeddings.</param>
		/// <param name="max_period">Controls the maximum frequency of the embeddings</param>
		/// <returns>torch.Tensor: an [N x dim] Tensor of positional embeddings.</returns>
		private static Tensor get_timestep_embedding(Tensor timesteps, int embedding_dim, bool flip_sin_to_cos = false, float downscale_freq_shift = 1, float scale = 1, int max_period = 10000)
		{
			using (NewDisposeScope())
			{
				if (timesteps.Dimensions != 1)
				{
					throw new ArgumentOutOfRangeException("Timesteps should be a 1d-array");
				}
				int half_dim = embedding_dim / 2;
				Tensor exponent = -Math.Log(max_period) * torch.arange(start: 0, stop: half_dim, dtype: torch.float32, device: timesteps.device);
				exponent = exponent / (half_dim - downscale_freq_shift);
				Tensor emb = torch.exp(exponent);
				emb = timesteps[.., TensorIndex.None].@float() * emb[TensorIndex.None, ..];

				// scale embeddings
				emb = scale * emb;

				// concat sine and cosine embeddings
				emb = torch.cat(new Tensor[] { torch.sin(emb), torch.cos(emb) }, dim: -1);

				// flip sine and cosine embeddings
				if (flip_sin_to_cos)
				{
					emb = torch.cat(new Tensor[] { emb[.., half_dim..], emb[.., ..half_dim] }, dim: -1);
				}

				// zero pad
				if (embedding_dim % 2 == 1)
				{
					emb = torch.nn.functional.pad(emb, (0, 1, 0, 0));
				}
				return emb.MoveToOuterDisposeScope();
			}
		}

	}
}
