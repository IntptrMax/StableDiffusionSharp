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

		internal class ViT_L_Clip : Module<Tensor, long, bool, Tensor>
		{
			private readonly CLIPTextModel transformer;

			public ViT_L_Clip(long n_vocab = 49408, long n_token = 77, long num_layers = 12, long n_heads = 12, long embed_dim = 768, long intermediate_size = 768 * 4) : base(nameof(ViT_L_Clip))
			{
				transformer = new CLIPTextModel(n_vocab, n_token, num_layers, n_heads, embed_dim, intermediate_size);
				RegisterComponents();
			}

			public override Tensor forward(Tensor token, long num_skip, bool with_final_ln)
			{
				return transformer.forward(token, num_skip, with_final_ln);
			}

			private class CLIPTextModel : Module<Tensor, long, bool, Tensor>
			{
				private readonly CLIPTextTransformer text_model;
				public CLIPTextModel(long n_vocab, long n_token, long num_layers, long n_heads, long embed_dim, long intermediate_size) : base(nameof(CLIPTextModel))
				{
					text_model = new CLIPTextTransformer(n_vocab, n_token, num_layers, n_heads, embed_dim, intermediate_size);
					RegisterComponents();
				}
				public override Tensor forward(Tensor x, long num_skip, bool with_final_ln)
				{
					return text_model.forward(x, num_skip, with_final_ln);
				}
			}

			private class CLIPTextTransformer : Module<Tensor, long, bool, Tensor>
			{
				private readonly CLIPTextEmbeddings embeddings;
				private readonly CLIPEncoder encoder;
				private readonly LayerNorm final_layer_norm;
				private readonly long num_layers;

				public CLIPTextTransformer(long n_vocab, long n_token, long num_layers, long n_heads, long embed_dim, long intermediate_size) : base(nameof(CLIPTextTransformer))
				{
					this.num_layers = num_layers;
					embeddings = new CLIPTextEmbeddings(n_vocab, embed_dim, n_token);
					encoder = new CLIPEncoder(num_layers, embed_dim, n_heads, intermediate_size, Activations.QuickGELU);
					final_layer_norm = LayerNorm(embed_dim);
					RegisterComponents();
				}
				public override Tensor forward(Tensor x, long num_skip, bool with_final_ln)
				{
					x = embeddings.forward(x);
					x = encoder.forward(x, num_skip);
					if (with_final_ln)
					{
						x = final_layer_norm.forward(x);
					}
					return x;
				}
			}

			private class CLIPTextEmbeddings : Module<Tensor, Tensor>
			{
				private readonly Embedding token_embedding;
				private readonly Embedding position_embedding;
				public CLIPTextEmbeddings(long n_vocab, long n_embd, long n_token) : base(nameof(CLIPTextEmbeddings))
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

			private class CLIPEncoderLayer : Module<Tensor, Tensor>
			{
				private readonly LayerNorm layer_norm1;
				private readonly LayerNorm layer_norm2;
				private readonly CLIPAttention self_attn;
				private readonly CLIPMLP mlp;

				public CLIPEncoderLayer(long n_head, long embed_dim, long intermediate_size, Activations activations = Activations.QuickGELU) : base(nameof(CLIPEncoderLayer))
				{
					layer_norm1 = LayerNorm(embed_dim);
					self_attn = new CLIPAttention(embed_dim, n_head);
					layer_norm2 = LayerNorm(embed_dim);
					mlp = new CLIPMLP(embed_dim, intermediate_size, embed_dim, activations);
					RegisterComponents();
				}

				public override Tensor forward(Tensor x)
				{
					x += this.self_attn.forward(this.layer_norm1.forward(x));
					x += this.mlp.forward(this.layer_norm2.forward(x));
					return x;
				}
			}

			private class CLIPMLP : Module<Tensor, Tensor>
			{
				private readonly Linear fc1;
				private readonly Linear fc2;
				private readonly Activations act_layer;
				public CLIPMLP(long in_features, long? hidden_features = null, long? out_features = null, Activations act_layer = Activations.QuickGELU, bool bias = true) : base(nameof(CLIPMLP))
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

			private class CLIPEncoder : Module<Tensor, long, Tensor>
			{
				private readonly Sequential layers;

				public CLIPEncoder(long num_layers, long embed_dim, long heads, long intermediate_size, Activations intermediate_activation) : base(nameof(CLIPEncoder))
				{
					layers = Sequential();
					for (int i = 0; i < num_layers; i++)
					{
						layers.append(new CLIPEncoderLayer(heads, embed_dim, intermediate_size, intermediate_activation));
					}
					RegisterComponents();
				}

				public override Tensor forward(Tensor x, long num_skip)
				{
					long num_act = num_skip > 0 ? (layers.Count - num_skip) : layers.Count;
					for (long i = 0; i < num_act; i++)
					{
						x = ((CLIPEncoderLayer)(layers.children().ToArray()[i])).forward(x);
					}
					//return layers.forward(x);
					return x;
				}
			}
		}

		private class ViT_bigG_Clip : Module<Tensor, long, bool, bool, Tensor>
		{
			private readonly int adm_in_channels;

			private readonly Embedding token_embedding;
			private readonly Parameter positional_embedding;
			private readonly Transformer transformer;
			private readonly LayerNorm ln_final;
			private readonly Parameter text_projection;

			public ViT_bigG_Clip(long n_vocab = 49408, long n_token = 77, long num_layers = 32, long n_heads = 20, long embed_dim = 1280, long intermediate_size = 1280 * 4, int adm_in_channels = 2816) : base(nameof(ViT_bigG_Clip))
			{
				token_embedding = Embedding(n_vocab, embed_dim);
				positional_embedding = Parameter(torch.zeros(size: [n_token, embed_dim]));
				text_projection = Parameter(torch.zeros(size: [embed_dim, embed_dim]));
				transformer = new Transformer(num_layers, embed_dim, n_heads, intermediate_size, Activations.GELU);
				ln_final = LayerNorm(embed_dim);
				this.adm_in_channels = adm_in_channels;
				RegisterComponents();
			}

			public override Tensor forward(Tensor x, long num_skip, bool with_final_ln, bool with_pool)
			{
				x = token_embedding.forward(x) + positional_embedding;
				x = transformer.forward(x, num_skip);
				if (with_final_ln || with_pool)
				{
					x = ln_final.forward(x);
				}
				if (with_pool)
				{
					x = x[.., -1, ..];
					x = torch.nn.functional.linear(x, text_projection);
					long padLength = this.adm_in_channels - x.shape[1];
					x = torch.nn.functional.pad(x, [0, padLength, 0, 0]);
					return x;
				}
				return x;
			}

			private class Transformer : Module<Tensor, long, Tensor>
			{
				private readonly Sequential resblocks;
				public Transformer(long num_layers, long embed_dim, long heads, long intermediate_size, Activations intermediate_activation) : base(nameof(Transformer))
				{
					resblocks = Sequential();
					for (int i = 0; i < num_layers; i++)
					{
						resblocks.append(new ResidualAttentionBlock(heads, embed_dim, intermediate_size, intermediate_activation));
					}
					RegisterComponents();
				}

				public override Tensor forward(Tensor x, long num_skip)
				{
					long num_act = num_skip > 0 ? (resblocks.Count - num_skip) : resblocks.Count;
					for (long i = 0; i < num_act; i++)
					{
						x = ((ResidualAttentionBlock)(resblocks.children().ToArray()[i])).forward(x);
					}
					//return resblocks.forward(x);
					return x;
				}
			}

			private class ResidualAttentionBlock : Module<Tensor, Tensor>
			{
				private readonly LayerNorm ln_1;
				private readonly LayerNorm ln_2;
				private readonly MultiheadAttention attn;
				private readonly Mlp mlp;

				public ResidualAttentionBlock(long n_head, long embed_dim, long intermediate_size, Activations activations = Activations.QuickGELU) : base(nameof(ResidualAttentionBlock))
				{
					ln_1 = LayerNorm(embed_dim);
					attn = new MultiheadAttention(embed_dim, n_head);
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

			private class MultiheadAttention : Module<Tensor, Tensor>
			{
				private readonly long heads;
				private readonly Parameter in_proj_weight;
				private readonly Parameter in_proj_bias;
				private readonly Linear out_proj;

				public MultiheadAttention(long embed_dim, long heads) : base(nameof(MultiheadAttention))
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

		internal class SDCliper : Module<Tensor, long, bool, Tensor>
		{
			internal readonly ViT_L_Clip cond_stage_model;
			public SDCliper(long n_vocab = 49408, long n_token = 77, long num_layers = 12, long n_heads = 12, long embed_dim = 768, long intermediate_size = 768 * 4) : base(nameof(SDCliper))
			{
				cond_stage_model = new ViT_L_Clip(n_vocab, n_token, num_layers, n_heads, embed_dim, intermediate_size);
				RegisterComponents();
			}
			public override Tensor forward(Tensor token, long num_skip, bool with_final_ln)
			{
				Device device = cond_stage_model.parameters().First().device;
				token = token.to(device);
				return cond_stage_model.forward(token, num_skip, with_final_ln);
			}
		}

		internal class SDXLCliper : Module<Tensor, (Tensor, Tensor)>
		{
			private readonly Embedders conditioner;
			public SDXLCliper(long n_vocab = 49408, long n_token = 77) : base(nameof(SDXLCliper))
			{
				conditioner = new Embedders();
				RegisterComponents();
			}

			public override (Tensor, Tensor) forward(Tensor token)
			{
				Device device = conditioner.parameters().First().device;
				token = token.to(device);
				return conditioner.forward(token);
			}

			private class Embedders : Module<Tensor, (Tensor, Tensor)>
			{
				private readonly ModuleList<Module> embedders;
				public Embedders() : base(nameof(Embedders))
				{
					Model model = new Model();
					embedders = ModuleList(new ViT_L_Clip(), model);
					RegisterComponents();
				}
				public override (Tensor, Tensor) forward(Tensor token)
				{
					using (NewDisposeScope())
					{
						Tensor vit_l_result = ((ViT_L_Clip)embedders[0]).forward(token, 2, false);
						Tensor vit_bigG_result = ((Model)embedders[1]).forward(token, 2, false, false);
						Tensor vit_bigG_vec = ((Model)embedders[1]).forward(token, 0, false, true);
						return (cat([vit_l_result, vit_bigG_result], 2).MoveToOuterDisposeScope(), vit_bigG_vec.MoveToOuterDisposeScope());
					}
				}
			}

			private class Model : Module<Tensor, long, bool, bool, Tensor>
			{
				private readonly ViT_bigG_Clip model;
				public Model() : base(nameof(Model))
				{
					model = new ViT_bigG_Clip();
					RegisterComponents();
				}
				public override Tensor forward(Tensor token, long num_skip, bool with_final_ln, bool with_pool)
				{
					return model.forward(token, num_skip, with_final_ln, with_pool);
				}
			}
		}

	}
}
