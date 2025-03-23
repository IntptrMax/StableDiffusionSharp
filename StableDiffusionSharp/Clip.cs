using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace StableDiffusionSharp
{
	internal class Clip
	{
		private enum Activations
		{
			ReLU,
			SiLU,
			QuickGELU,
			GELU
		}

		internal class ViT_L_Clip : Module<Tensor, Tensor>
		{
			internal readonly Sequential transformer;
			public ViT_L_Clip(long n_vocab = 49408, long n_token = 77, long num_layers = 12, long n_heads = 12, long embed_dim = 768, long intermediate_size = 768 * 4) : base(nameof(ViT_L_Clip))
			{
				transformer = Sequential(("text_model", Sequential(
					[("embeddings", new CLIPEmbedding(n_vocab, embed_dim, n_token)),
					("encoder", new ClipEncoder(num_layers, embed_dim, n_heads, intermediate_size, Activations.QuickGELU)),
					("final_layer_norm", LayerNorm(embed_dim))])));
				RegisterComponents();
			}

			public override Tensor forward(Tensor token)
			{
				return transformer.forward(token);
			}


			private class CLIPEmbedding : Module<Tensor, Tensor>
			{
				private readonly Embedding token_embedding;
				private readonly Embedding position_embedding;
				public CLIPEmbedding(long n_vocab, long n_embd, long n_token) : base(nameof(CLIPEmbedding))
				{
					token_embedding = Embedding(n_vocab, n_embd);
					position_embedding = Embedding(n_token, n_embd);
					RegisterComponents();
				}

				public override Tensor forward(Tensor tokens)
				{
					return token_embedding.forward(tokens) + position_embedding.weight!;
				}
			}

			private class CLIPLayer : Module<Tensor, Tensor>
			{
				private readonly LayerNorm layer_norm1;
				private readonly LayerNorm layer_norm2;
				private readonly CLIPAttention self_attn;
				private readonly Mlp mlp;

				public CLIPLayer(long n_head, long embed_dim, long intermediate_size, Activations activations = Activations.QuickGELU) : base(nameof(CLIPLayer))
				{
					layer_norm1 = LayerNorm(embed_dim);
					self_attn = new CLIPAttention(embed_dim, n_head);
					layer_norm2 = LayerNorm(embed_dim);
					mlp = new Mlp(embed_dim, intermediate_size, embed_dim, activations);
					RegisterComponents();
				}

				public override Tensor forward(Tensor x)
				{
					x += this.self_attn.forward(this.layer_norm1.forward(x));
					x += this.mlp.forward(this.layer_norm2.forward(x));
					return x;
				}
			}



			private class Mlp : Module<Tensor, Tensor>
			{
				private readonly Linear fc1;
				private readonly Linear fc2;
				private readonly Activations act_layer;
				public Mlp(long in_features, long? hidden_features = null, long? out_features = null, Activations act_layer = Activations.QuickGELU, bool bias = true) : base(nameof(Mlp))
				{
					out_features ??= in_features;
					hidden_features ??= out_features;

					this.fc1 = Linear(in_features, (long)hidden_features, hasBias: bias);
					this.fc2 = Linear((long)hidden_features, (long)out_features, hasBias: bias);
					this.act_layer = act_layer;
					RegisterComponents();
				}

				public override Tensor forward(Tensor x)
				{
					x = this.fc1.forward(x);

					switch (this.act_layer)
					{
						case Activations.ReLU:
							x = torch.nn.functional.relu(x);
							break;
						case Activations.SiLU:
							x = torch.nn.functional.silu(x);
							break;
						case Activations.QuickGELU:
							x = x * torch.sigmoid(1.702 * x);
							break;
						case Activations.GELU:
							x = torch.nn.functional.gelu(x);
							break;
					}
					x = this.fc2.forward(x);
					return x;
				}
			}

			private class CLIPAttention : Module<Tensor, Tensor>
			{
				private readonly long heads;
				private readonly Linear q_proj;
				private readonly Linear k_proj;
				private readonly Linear v_proj;
				private readonly Linear out_proj;

				public CLIPAttention(long embed_dim, long heads) : base(nameof(CLIPAttention))
				{
					this.heads = heads;
					this.q_proj = nn.Linear(embed_dim, embed_dim, hasBias: true);
					this.k_proj = nn.Linear(embed_dim, embed_dim, hasBias: true);
					this.v_proj = nn.Linear(embed_dim, embed_dim, hasBias: true);
					this.out_proj = nn.Linear(embed_dim, embed_dim, hasBias: true);

					RegisterComponents();
				}

				public override Tensor forward(Tensor x)
				{
					using (var _ = NewDisposeScope())
					{
						Tensor q = this.q_proj.forward(x);
						Tensor k = this.k_proj.forward(x);
						Tensor v = this.v_proj.forward(x);
						Tensor output = attention(q, k, v, this.heads);
						//TensorInfo output = self_atten(to_q, to_k, to_v, this.heads);
						return this.out_proj.forward(output).MoveToOuterDisposeScope();
					}
				}

				private static Tensor self_atten(Tensor q, Tensor k, Tensor v, long heads)
				{
					long[] input_shape = q.shape;
					long batch_size = q.shape[0];
					long sequence_length = q.shape[1];
					long d_head = q.shape[2] / heads;
					long[] interim_shape = new long[] { batch_size, sequence_length, heads, d_head };

					q = q.view(interim_shape).transpose(1, 2);
					k = k.view(interim_shape).transpose(1, 2);
					v = v.view(interim_shape).transpose(1, 2);

					var weight = torch.matmul(q, k.transpose(-1, -2));
					var mask = torch.ones_like(weight).triu(1).to(torch.@bool);
					weight.masked_fill_(mask, Single.NegativeInfinity);

					weight = weight / (float)Math.Sqrt(d_head);
					weight = torch.nn.functional.softmax(weight, dim: -1);

					var output = torch.matmul(weight, v);
					output = output.transpose(1, 2);
					output = output.reshape(input_shape);
					return output;
				}

				// Convenience wrapper around a basic attention operation
				private static Tensor attention(Tensor q, Tensor k, Tensor v, long heads)
				{
					long b = q.shape[0];
					long dim_head = q.shape[2];
					dim_head /= heads;
					q = q.view(b, -1, heads, dim_head).transpose(1, 2);
					k = k.view(b, -1, heads, dim_head).transpose(1, 2);
					v = v.view(b, -1, heads, dim_head).transpose(1, 2);
					Tensor output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_casual: true);
					output = output.transpose(1, 2);
					output = output.view(b, -1, heads * dim_head);
					return output;
				}
			}

			private class ClipEncoder : Module<Tensor, Tensor>
			{
				private readonly Sequential layers;

				public ClipEncoder(long num_layers, long embed_dim, long heads, long intermediate_size, Activations intermediate_activation) : base(nameof(ClipEncoder))
				{
					layers = Sequential();
					for (int i = 0; i < num_layers; i++)
					{
						layers.append(new CLIPLayer(heads, embed_dim, intermediate_size, intermediate_activation));
					}
					RegisterComponents();
				}

				public override Tensor forward(Tensor x)
				{
					return layers.forward(x);
				}
			}
		}

		private class ViT_bigG_Clip : Module<Tensor, Tensor>
		{
			private readonly Embedding token_embedding;
			private readonly Parameter positional_embedding;
			private readonly ClipEncoder transformer;
			private readonly LayerNorm ln_final;

			public ViT_bigG_Clip(long n_vocab = 49408, long n_token = 77, long num_layers = 32, long n_heads = 20, long embed_dim = 1280, long intermediate_size = 1280 * 4) : base(nameof(ViT_L_Clip))
			{
				token_embedding = Embedding(n_vocab, embed_dim);
				positional_embedding = Parameter(torch.zeros(size: [n_token, embed_dim]));
				transformer = new ClipEncoder(num_layers, embed_dim, n_heads, intermediate_size, Activations.GELU);
				ln_final = LayerNorm(embed_dim);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				x = token_embedding.forward(x) + positional_embedding;
				x = transformer.forward(x);
				x = ln_final.forward(x);
				return x;
			}

			private class ClipEncoder : Module<Tensor, Tensor>
			{
				private readonly Sequential resblocks;
				public ClipEncoder(long num_layers, long embed_dim, long heads, long intermediate_size, Activations intermediate_activation) : base(nameof(ClipEncoder))
				{
					resblocks = Sequential();
					for (int i = 0; i < num_layers; i++)
					{
						resblocks.append(new CLIPLayer(heads, embed_dim, intermediate_size, intermediate_activation));
					}
					RegisterComponents();
				}

				public override Tensor forward(Tensor x)
				{
					return resblocks.forward(x);
				}
			}

			private class CLIPLayer : Module<Tensor, Tensor>
			{
				private readonly LayerNorm ln_1;
				private readonly LayerNorm ln_2;
				private readonly CLIPAttention attn;
				private readonly Mlp mlp;

				public CLIPLayer(long n_head, long embed_dim, long intermediate_size, Activations activations = Activations.QuickGELU) : base(nameof(CLIPLayer))
				{
					ln_1 = LayerNorm(embed_dim);
					attn = new CLIPAttention(embed_dim, n_head);
					ln_2 = LayerNorm(embed_dim);
					mlp = new Mlp(embed_dim, intermediate_size, embed_dim, activations);
					RegisterComponents();
				}

				public override Tensor forward(Tensor x)
				{
					x += this.attn.forward(this.ln_1.forward(x));
					x += this.mlp.forward(this.ln_2.forward(x));
					return x;
				}
			}

			private class Mlp : Module<Tensor, Tensor>
			{
				private readonly Linear c_fc;
				private readonly Linear c_proj;
				private readonly Activations act_layer;
				public Mlp(long in_features, long? hidden_features = null, long? out_features = null, Activations act_layer = Activations.QuickGELU, bool bias = true) : base(nameof(Mlp))
				{
					out_features ??= in_features;
					hidden_features ??= out_features;

					this.c_fc = Linear(in_features, (long)hidden_features, hasBias: bias);
					this.c_proj = Linear((long)hidden_features, (long)out_features, hasBias: bias);
					this.act_layer = act_layer;
					RegisterComponents();
				}

				public override Tensor forward(Tensor x)
				{
					x = this.c_fc.forward(x);

					switch (this.act_layer)
					{
						case Activations.ReLU:
							x = torch.nn.functional.relu(x);
							break;
						case Activations.SiLU:
							x = torch.nn.functional.silu(x);
							break;
						case Activations.QuickGELU:
							x = x * torch.sigmoid(1.702 * x);
							break;
						case Activations.GELU:
							x = torch.nn.functional.gelu(x);
							break;
					}
					x = this.c_proj.forward(x);
					return x;
				}
			}

			private class CLIPAttention : Module<Tensor, Tensor>
			{
				private readonly long heads;
				private readonly Parameter in_proj_weight;
				private readonly Parameter in_proj_bias;
				private readonly Linear out_proj;

				public CLIPAttention(long embed_dim, long heads) : base(nameof(CLIPAttention))
				{
					this.heads = heads;
					this.in_proj_weight = Parameter(torch.zeros([3 * embed_dim, embed_dim]));
					this.in_proj_bias = Parameter(torch.zeros([3 * embed_dim]));
					this.out_proj = nn.Linear(embed_dim, embed_dim, hasBias: true);

					RegisterComponents();
				}

				public override Tensor forward(Tensor x)
				{
					using (var _ = NewDisposeScope())
					{
						Tensor[] qkv = torch.nn.functional.linear(x, this.in_proj_weight, this.in_proj_bias).chunk(3, 2);
						Tensor q = qkv[0];
						Tensor k = qkv[1];
						Tensor v = qkv[2];
						Tensor output = attention(q, k, v, this.heads);
						//TensorInfo output = self_atten(to_q, to_k, to_v, this.heads);
						return this.out_proj.forward(output).MoveToOuterDisposeScope();
					}
				}

				private static Tensor self_atten(Tensor q, Tensor k, Tensor v, long heads)
				{
					long[] input_shape = q.shape;
					long batch_size = q.shape[0];
					long sequence_length = q.shape[1];
					long d_head = q.shape[2] / heads;
					long[] interim_shape = new long[] { batch_size, sequence_length, heads, d_head };

					q = q.view(interim_shape).transpose(1, 2);
					k = k.view(interim_shape).transpose(1, 2);
					v = v.view(interim_shape).transpose(1, 2);

					var weight = torch.matmul(q, k.transpose(-1, -2));
					var mask = torch.ones_like(weight).triu(1).to(torch.@bool);
					weight.masked_fill_(mask, Single.NegativeInfinity);

					weight = weight / (float)Math.Sqrt(d_head);
					weight = torch.nn.functional.softmax(weight, dim: -1);

					var output = torch.matmul(weight, v);
					output = output.transpose(1, 2);
					output = output.reshape(input_shape);
					return output;
				}

				// Convenience wrapper around a basic attention operation
				private static Tensor attention(Tensor q, Tensor k, Tensor v, long heads)
				{
					long b = q.shape[0];
					long dim_head = q.shape[2];
					dim_head /= heads;
					q = q.view(b, -1, heads, dim_head).transpose(1, 2);
					k = k.view(b, -1, heads, dim_head).transpose(1, 2);
					v = v.view(b, -1, heads, dim_head).transpose(1, 2);
					Tensor output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_casual: true);
					output = output.transpose(1, 2);
					output = output.view(b, -1, heads * dim_head);
					return output;
				}
			}
		}

		internal class SDCliper : Module<Tensor, Tensor>
		{
			internal readonly ViT_L_Clip cond_stage_model;
			public SDCliper(long n_vocab = 49408, long n_token = 77, long num_layers = 12, long n_heads = 12, long embed_dim = 768, long intermediate_size = 768 * 4) : base(nameof(SDCliper))
			{
				cond_stage_model = new ViT_L_Clip(n_vocab, n_token, num_layers, n_heads, embed_dim, intermediate_size);
				RegisterComponents();
			}
			public override Tensor forward(Tensor token)
			{
				return cond_stage_model.forward(token);
			}
		}

		internal class SDXLCliper : Module<Tensor, Tensor>
		{
			private readonly Embedders conditioner;
			public SDXLCliper(long n_vocab = 49408, long n_token = 77) : base(nameof(SDXLCliper))
			{
				conditioner = new Embedders();
				RegisterComponents();
			}

			public override Tensor forward(Tensor token)
			{
				return conditioner.forward(token);
			}

			private class Embedders : Module<Tensor, Tensor>
			{
				private readonly Sequential embedders;
				public Embedders() : base(nameof(Embedders))
				{
					embedders = Sequential(new ViT_L_Clip(), Sequential(("model", new ViT_bigG_Clip())));
					RegisterComponents();
				}
				public override Tensor forward(Tensor token)
				{
					using (NewDisposeScope())
					{
						Tensor vit_l_result = embedders[0].call(token);
						Tensor vit_bigG_result = embedders[1].call(token);
						return (cat([vit_l_result, vit_bigG_result], 2).MoveToOuterDisposeScope());
					}

				}
			}
		}

	}
}
