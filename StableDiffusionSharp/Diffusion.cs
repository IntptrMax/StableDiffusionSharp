using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace StableDiffusionSharp
{
	internal class Diffusion : Module<Tensor, Tensor, Tensor, Tensor>
	{
		private class SelfAttention : Module<Tensor, Tensor>
		{
			private readonly Linear to_q;
			private readonly Linear to_k;
			private readonly Linear to_v;
			private readonly Sequential to_out;
			private readonly long n_heads_;
			private readonly long d_head;
			bool causal_mask_;
			float dropout_p;

			public SelfAttention(long channels, long n_heads, bool in_proj_bias = false, bool out_proj_bias = true, float dropout_p = 0.1f, bool causal_mask = false) : base("SelfAttention")
			{
				this.to_q = Linear(channels, channels, hasBias: in_proj_bias);
				this.to_k = Linear(channels, channels, hasBias: in_proj_bias);
				this.to_v = Linear(channels, channels, hasBias: in_proj_bias);
				this.to_out = Sequential(Linear(channels, channels, hasBias: out_proj_bias));
				this.n_heads_ = n_heads;
				this.d_head = channels / n_heads;
				this.causal_mask_ = causal_mask;
				this.dropout_p = dropout_p;
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using (NewDisposeScope())
				{
					long[] input_shape = x.shape;
					long batch_size = input_shape[0];
					long sequence_length = input_shape[1];
					Tensor q = to_q.forward(x);
					Tensor k = to_k.forward(x);
					Tensor v = to_v.forward(x);

					q = q.view([batch_size, sequence_length, n_heads_, d_head]).transpose(1, 2).contiguous();
					k = k.view([batch_size, sequence_length, n_heads_, d_head]).transpose(1, 2).contiguous();
					v = v.view([batch_size, sequence_length, n_heads_, d_head]).transpose(1, 2).contiguous();
					Tensor output = torch.nn.functional.scaled_dot_product_attention(q, k, v, p: dropout_p, is_casual: causal_mask_);
					output = output.transpose(1, 2).reshape(input_shape).contiguous();
					output = to_out.forward(output);
					return output.MoveToOuterDisposeScope();
				}
			}
		}

		private class CrossAttention : Module<Tensor, Tensor, Tensor>
		{
			private readonly Linear to_q;
			private readonly Linear to_k;
			private readonly Linear to_v;
			private readonly Sequential to_out;
			private readonly long n_heads_;
			private readonly long d_head;
			private readonly bool causal_mask_;
			private readonly float dropout_p;

			public CrossAttention(long channels, long n_heads, long d_cross, bool in_proj_bias = false, bool out_proj_bias = true, float dropout_p = 0.2f, bool causal_mask = true) : base("CrossAttention")
			{
				to_q = Linear(channels, channels, hasBias: in_proj_bias);
				to_k = Linear(d_cross, channels, hasBias: in_proj_bias);
				to_v = Linear(d_cross, channels, hasBias: in_proj_bias);
				to_out = Sequential(Linear(channels, channels, hasBias: out_proj_bias));
				n_heads_ = n_heads;
				d_head = channels / n_heads;
				this.causal_mask_ = causal_mask;
				this.dropout_p = dropout_p;
				RegisterComponents();
			}

			public override Tensor forward(Tensor x, Tensor y)
			{
				using (NewDisposeScope())
				{
					long[] input_shape = x.shape;
					long batch_size = input_shape[0];
					long sequence_length = input_shape[1];

					long[] interim_shape = [batch_size, -1, n_heads_, d_head];
					Tensor q = to_q.forward(x);
					Tensor k = to_k.forward(y);
					Tensor v = to_v.forward(y);

					q = q.view(interim_shape).transpose(1, 2);
					k = k.view(interim_shape).transpose(1, 2);
					v = v.view(interim_shape).transpose(1, 2);

					Tensor output = torch.nn.functional.scaled_dot_product_attention(q, k, v, p: 0, is_casual: causal_mask_);
					output = output.transpose(1, 2).reshape(input_shape);
					output = to_out.forward(output);
					return output.MoveToOuterDisposeScope();
				}
			}
		}

		private class ResnetBlock : Module<Tensor, Tensor, Tensor>
		{
			private readonly int in_channels;
			private readonly int out_channels;

			private readonly Module<Tensor, Tensor> skip_connection;
			private readonly Sequential emb_layers;
			private readonly Sequential in_layers;
			private readonly Sequential out_layers;

			public ResnetBlock(int in_channels, int out_channels, double dropout = 0.0, int temb_channels = 1280) : base(nameof(ResnetBlock))
			{
				this.in_channels = in_channels;
				out_channels = out_channels < 1 ? in_channels : out_channels;
				this.out_channels = out_channels;

				this.in_layers = Sequential(GroupNorm(32, in_channels), SiLU(), Conv2d(in_channels, out_channels, kernel_size: 3, stride: 1, padding: 1));

				if (temb_channels > 0)
				{
					this.emb_layers = Sequential(SiLU(), Linear(temb_channels, out_channels));
				}

				this.out_layers = Sequential(GroupNorm(32, out_channels), SiLU(), Dropout(dropout), Conv2d(out_channels, out_channels, kernel_size: 3, stride: 1, padding: 1));

				if (this.in_channels != this.out_channels)
				{
					this.skip_connection = torch.nn.Conv2d(in_channels: in_channels, out_channels: this.out_channels, kernel_size: 1);
				}
				else
				{
					this.skip_connection = Identity();
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
						time = this.emb_layers.forward(time);
						hidden = hidden + time.unsqueeze(-1).unsqueeze(-1);
					}

					hidden = out_layers.forward(hidden);
					if (this.in_channels != this.out_channels)
					{
						x = this.skip_connection.forward(x);
					}
					return (x + hidden).MoveToOuterDisposeScope();
				}
			}
		}

		private class TransformerBlock : Module<Tensor, Tensor, Tensor>
		{
			private LayerNorm norm1;
			private SelfAttention attn1;
			private LayerNorm norm2;
			private CrossAttention attn2;
			private LayerNorm norm3;
			private FeedForward ff;

			public TransformerBlock(int channels, int n_head = 8, int n_cross = 768) : base(nameof(TransformerBlock))
			{
				this.norm1 = LayerNorm(channels);
				this.attn1 = new SelfAttention(channels, n_head);
				this.norm2 = LayerNorm(channels);
				this.attn2 = new CrossAttention(channels, n_head, n_cross);
				this.norm3 = LayerNorm(channels);
				this.ff = new FeedForward(channels, glu: true);
				RegisterComponents();
			}
			public override Tensor forward(Tensor x, Tensor context)
			{
				var residue_short = x;
				x = norm1.forward(x);
				x = attn1.forward(x);
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

		private class AttenGroup : Module<Tensor, Tensor, Tensor>
		{
			private readonly GroupNorm norm;
			private readonly Conv2d proj_in;
			private readonly Conv2d proj_out;
			private readonly ModuleList<Module<Tensor, Tensor, Tensor>> transformer_blocks;

			public AttenGroup(int channels, int n_head = 8, int n_cross = 768, float drop_out = 0.0f) : base(nameof(AttenGroup))
			{
				this.norm = GroupNorm(32, channels);
				this.proj_in = Conv2d(channels, channels, kernel_size: 1);
				this.proj_out = Conv2d(channels, channels, kernel_size: 1);
				this.transformer_blocks = new ModuleList<Module<Tensor, Tensor, Tensor>>(new TransformerBlock(channels, n_head, n_cross));
				RegisterComponents();
			}

			public override Tensor forward(Tensor x, Tensor context)
			{
				using (NewDisposeScope())
				{
					Tensor residue_short = x;
					x = norm.forward(x);
					x = proj_in.forward(x);

					var n = x.shape[0];
					var c = x.shape[1];
					var h = x.shape[2];
					var w = x.shape[3];
					x = x.view([n, c, h * w]);
					x = x.transpose(-1, -2);

					foreach (Module<Tensor, Tensor, Tensor> layer in transformer_blocks.children())
					{
						x = layer.forward(x, context);
					}

					x = x.transpose(-1, -2);
					x = x.view([n, c, h, w]);

					residue_short = residue_short + proj_out.forward(x);
					return residue_short.MoveToOuterDisposeScope();
				}
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
				if (this.with_conv && this.conv is not null)
				{
					output = this.conv.forward(output);
				}
				return output;
			}
		}

		private class SwitchSequential : Sequential<Tensor, Tensor, Tensor, Tensor>
		{
			internal SwitchSequential(params (string name, torch.nn.Module)[] modules) : base(modules)
			{
				RegisterComponents();
			}

			internal SwitchSequential(params torch.nn.Module[] modules) : base(modules)
			{
				RegisterComponents();
			}

			public override torch.Tensor forward(torch.Tensor x, torch.Tensor context, torch.Tensor time)
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
							case AttenGroup abl:
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

		private class GEGLU : Module<Tensor, Tensor>
		{
			private readonly Linear proj;
			public GEGLU(int dim_in, int dim_out) : base(nameof(GEGLU))
			{
				this.proj = nn.Linear(dim_in, dim_out * 2);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				Tensor[] result = this.proj.forward(x).chunk(2, dim: -1);
				x = result[0];
				Tensor gate = result[1];
				return x * torch.nn.functional.gelu(gate);
			}
		}

		private class FeedForward : Module<Tensor, Tensor>
		{
			private readonly Sequential net;

			public FeedForward(int dim, int? dim_out = null, int mult = 4, bool glu = true, float dropout = 0.0f) : base(nameof(FeedForward))
			{
				int inner_dim = dim * mult;
				int dim_ot = dim_out ?? dim;
				Module<Tensor, Tensor> project_in = glu ? new GEGLU(dim, inner_dim) : Sequential(nn.Linear(dim, inner_dim), nn.GELU());
				this.net = Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_ot));
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				return this.net.forward(input);
			}
		}

		private class UNet : Module<Tensor, Tensor, Tensor, Tensor>
		{
			private readonly int ch;
			private readonly int time_embed_dim;
			private readonly int in_channels;
			private readonly bool use_timestep;

			private readonly Sequential time_embed;
			private readonly ModuleList<SwitchSequential> input_blocks;
			private readonly SwitchSequential middle_block;
			private readonly ModuleList<SwitchSequential> output_blocks;
			private readonly Sequential @out;

			public UNet(int model_channels, int in_channels, int[]? channel_mult = null, int num_res_blocks = 2, int context_dim = 768, int num_heads = 8, float dropout = 0.0f, bool use_timestep = true) : base(nameof(UNet))
			{
				channel_mult = channel_mult ?? [1, 2, 4, 4];

				this.ch = model_channels;
				this.time_embed_dim = model_channels * 4;
				this.in_channels = in_channels;
				this.use_timestep = use_timestep;

				List<int> input_block_channels = [model_channels];

				if (use_timestep)
				{
					// timestep embedding
					this.time_embed = Sequential([torch.nn.Linear(model_channels, time_embed_dim), SiLU(), torch.nn.Linear(time_embed_dim, time_embed_dim)]);
				}

				// downsampling
				this.input_blocks = new ModuleList<SwitchSequential>();
				input_blocks.Add(new SwitchSequential(Conv2d(in_channels, this.ch, kernel_size: 3, padding: 1)));

				for (int i = 0; i < channel_mult.Length; i++)
				{
					int in_ch = model_channels * channel_mult[i > 0 ? i - 1 : i];
					int out_ch = model_channels * channel_mult[i];

					for (int j = 0; j < num_res_blocks; j++)
					{
						input_blocks.Add(new SwitchSequential(new ResnetBlock(in_ch, out_ch, dropout, time_embed_dim), (i < channel_mult.Length - 1) ? new AttenGroup(out_ch, num_heads, context_dim, dropout) : Identity()));
						input_block_channels.Add(in_ch);
						in_ch = out_ch;
					}
					if (i < channel_mult.Length - 1)
					{
						input_blocks.Add(new SwitchSequential(Sequential(("op", Conv2d(out_ch, out_ch, 3, stride: 2, padding: 1)))));
						input_block_channels.Add(out_ch);
					}
				}

				// middle block
				middle_block = new SwitchSequential(new ResnetBlock(time_embed_dim, time_embed_dim, dropout, time_embed_dim), new AttenGroup(time_embed_dim, num_heads, context_dim, dropout), new ResnetBlock(1280, 1280));

				var reversed_mult = channel_mult.Reverse().ToList();
				int prev_channels = time_embed_dim;
				this.output_blocks = new ModuleList<SwitchSequential>();
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
							output_blocks.Add(new SwitchSequential(
								new ResnetBlock(input_channels, current_channels, dropout, time_embed_dim),
								new AttenGroup(current_channels, num_heads, context_dim, dropout),
								(has_upsample ? new Upsample(current_channels) : Identity())));
						}
						else
						{
							output_blocks.Add(new SwitchSequential(
								new ResnetBlock(input_channels, current_channels, dropout, time_embed_dim),
								(has_upsample ? new Upsample(current_channels) : Identity())));
						}

						prev_channels = current_channels;
					}
				}

				@out = Sequential(GroupNorm(32, model_channels), SiLU(), Conv2d(model_channels, in_channels, kernel_size: 3, padding: 1));

				RegisterComponents();

			}
			public override Tensor forward(Tensor x, Tensor context, Tensor time)
			{
				using var _ = NewDisposeScope();
				time = time_embed.forward(time);

				List<Tensor> skip_connections = new List<Tensor>();
				foreach (SwitchSequential layers in input_blocks)
				{
					x = layers.forward(x, context, time);
					skip_connections.Add(x);
				}
				x = middle_block.forward(x, context, time);
				foreach (SwitchSequential layers in output_blocks)
				{
					Tensor index = skip_connections.Last();
					x = torch.cat([x, index], 1);
					skip_connections.RemoveAt(skip_connections.Count - 1);
					x = layers.forward(x, context, time);
				}

				x = @out.forward(x);
				return x.MoveToOuterDisposeScope();
			}
		}


		private readonly ModuleList<Module<Tensor, Tensor, Tensor, Tensor>> model;

		public Diffusion(int model_channels, int in_channels, int num_heads = 8, int context_dim = 768, float dropout = 0.0f, bool use_timestep = true) : base(nameof(Diffusion))
		{
			model = new ModuleList<Module<Tensor, Tensor, Tensor, Tensor>>();
			model.add_module("diffusion_model", new UNet(model_channels, in_channels, context_dim: context_dim, num_heads: num_heads, dropout: dropout, use_timestep: use_timestep));
			RegisterComponents();
		}

		public override Tensor forward(Tensor latent, Tensor context, Tensor time)
		{
			using (NewDisposeScope())
			{
				foreach (var layer in model.children())
				{
					latent = ((Module<Tensor, Tensor, Tensor, Tensor>)layer).forward(latent, context, time);
				}
				return latent.MoveToOuterDisposeScope();
			}

		}

	}
}
