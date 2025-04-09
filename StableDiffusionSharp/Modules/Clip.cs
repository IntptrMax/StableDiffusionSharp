using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace StableDiffusionSharp.Modules
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

			public ViT_L_Clip(long n_vocab = 49408, long n_token = 77, long num_layers = 12, long n_heads = 12, long embed_dim = 768, long intermediate_size = 768 * 4, Device? device = null, ScalarType? dtype = null) : base(nameof(ViT_L_Clip))
			{
				transformer = new CLIPTextModel(n_vocab, n_token, num_layers, n_heads, embed_dim, intermediate_size, device: device, dtype: dtype);
				RegisterComponents();
			}

			public override Tensor forward(Tensor token, long num_skip, bool with_final_ln)
			{
				Device device = transformer.parameters().First().device;
				token = token.to(device);
				return transformer.forward(token, num_skip, with_final_ln);
			}

			private class CLIPTextModel : Module<Tensor, long, bool, Tensor>
			{
				private readonly CLIPTextTransformer text_model;
				public CLIPTextModel(long n_vocab, long n_token, long num_layers, long n_heads, long embed_dim, long intermediate_size, Device? device = null, ScalarType? dtype = null) : base(nameof(CLIPTextModel))
				{
					text_model = new CLIPTextTransformer(n_vocab, n_token, num_layers, n_heads, embed_dim, intermediate_size, device: device, dtype: dtype);
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

				public CLIPTextTransformer(long n_vocab, long n_token, long num_layers, long n_heads, long embed_dim, long intermediate_size, Device? device = null, ScalarType? dtype = null) : base(nameof(CLIPTextTransformer))
				{
					this.num_layers = num_layers;
					embeddings = new CLIPTextEmbeddings(n_vocab, embed_dim, n_token, device: device, dtype: dtype);
					encoder = new CLIPEncoder(num_layers, embed_dim, n_heads, intermediate_size, Activations.QuickGELU, device: device, dtype: dtype);
					final_layer_norm = LayerNorm(embed_dim, device: device, dtype: dtype);
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
				private readonly Parameter position_ids;
				public CLIPTextEmbeddings(long n_vocab, long n_embd, long n_token, Device? device = null, ScalarType? dtype = null) : base(nameof(CLIPTextEmbeddings))
				{
					position_ids = Parameter(zeros(size: [1, n_token], device: device, dtype: dtype));
					token_embedding = Embedding(n_vocab, n_embd, device: device, dtype: dtype);
					position_embedding = Embedding(n_token, n_embd, device: device, dtype: dtype);
					RegisterComponents();
				}

				public override Tensor forward(Tensor tokens)
				{
					return token_embedding.forward(tokens) + position_embedding.forward(position_ids.@long());
				}
			}

			private class CLIPEncoderLayer : Module<Tensor, Tensor>
			{
				private readonly LayerNorm layer_norm1;
				private readonly LayerNorm layer_norm2;
				private readonly CLIPAttention self_attn;
				private readonly CLIPMLP mlp;

				public CLIPEncoderLayer(long n_head, long embed_dim, long intermediate_size, Activations activations = Activations.QuickGELU, Device? device = null, ScalarType? dtype = null) : base(nameof(CLIPEncoderLayer))
				{
					layer_norm1 = LayerNorm(embed_dim, device: device, dtype: dtype);
					self_attn = new CLIPAttention(embed_dim, n_head, device: device, dtype: dtype);
					layer_norm2 = LayerNorm(embed_dim, device: device, dtype: dtype);
					mlp = new CLIPMLP(embed_dim, intermediate_size, embed_dim, activations, device: device, dtype: dtype);
					RegisterComponents();
				}

				public override Tensor forward(Tensor x)
				{
					x += self_attn.forward(layer_norm1.forward(x));
					x += mlp.forward(layer_norm2.forward(x));
					return x;
				}
			}

			private class CLIPMLP : Module<Tensor, Tensor>
			{
				private readonly Linear fc1;
				private readonly Linear fc2;
				private readonly Activations act_layer;
				public CLIPMLP(long in_features, long? hidden_features = null, long? out_features = null, Activations act_layer = Activations.QuickGELU, bool bias = true, Device? device = null, ScalarType? dtype = null) : base(nameof(CLIPMLP))
				{
					out_features ??= in_features;
					hidden_features ??= out_features;

					fc1 = Linear(in_features, (long)hidden_features, hasBias: bias, device: device, dtype: dtype);
					fc2 = Linear((long)hidden_features, (long)out_features, hasBias: bias, device: device, dtype: dtype);
					this.act_layer = act_layer;
					RegisterComponents();
				}

				public override Tensor forward(Tensor x)
				{
					x = fc1.forward(x);

					switch (act_layer)
					{
						case Activations.ReLU:
							x = functional.relu(x);
							break;
						case Activations.SiLU:
							x = functional.silu(x);
							break;
						case Activations.QuickGELU:
							x = x * sigmoid(1.702 * x);
							break;
						case Activations.GELU:
							x = functional.gelu(x);
							break;
					}
					x = fc2.forward(x);
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

				public CLIPAttention(long embed_dim, long heads, Device? device = null, ScalarType? dtype = null) : base(nameof(CLIPAttention))
				{
					this.heads = heads;
					q_proj = Linear(embed_dim, embed_dim, hasBias: true, device: device, dtype: dtype);
					k_proj = Linear(embed_dim, embed_dim, hasBias: true, device: device, dtype: dtype);
					v_proj = Linear(embed_dim, embed_dim, hasBias: true, device: device, dtype: dtype);
					out_proj = Linear(embed_dim, embed_dim, hasBias: true, device: device, dtype: dtype);

					RegisterComponents();
				}

				public override Tensor forward(Tensor x)
				{
					using (var _ = NewDisposeScope())
					{
						Tensor q = q_proj.forward(x);
						Tensor k = k_proj.forward(x);
						Tensor v = v_proj.forward(x);
						Tensor output = attention(q, k, v, heads);
						//TensorInfo output = self_atten(to_q, to_k, to_v, this.heads);
						return out_proj.forward(output).MoveToOuterDisposeScope();
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

					var weight = matmul(q, k.transpose(-1, -2));
					var mask = ones_like(weight).triu(1).to(@bool);
					weight.masked_fill_(mask, float.NegativeInfinity);

					weight = weight / (float)Math.Sqrt(d_head);
					weight = functional.softmax(weight, dim: -1);

					var output = matmul(weight, v);
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
					Tensor output = functional.scaled_dot_product_attention(q, k, v, is_casual: true);
					output = output.transpose(1, 2);
					output = output.view(b, -1, heads * dim_head);
					return output;
				}
			}

			private class CLIPEncoder : Module<Tensor, long, Tensor>
			{
				private readonly ModuleList<CLIPEncoderLayer> layers;

				public CLIPEncoder(long num_layers, long embed_dim, long heads, long intermediate_size, Activations intermediate_activation, Device? device = null, ScalarType? dtype = null) : base(nameof(CLIPEncoder))
				{
					layers = new ModuleList<CLIPEncoderLayer>();
					for (int i = 0; i < num_layers; i++)
					{
						layers.append(new CLIPEncoderLayer(heads, embed_dim, intermediate_size, intermediate_activation, device: device, dtype: dtype));
					}
					RegisterComponents();
				}

				public override Tensor forward(Tensor x, long num_skip)
				{
					long num_act = num_skip > 0 ? layers.Count - num_skip : layers.Count;
					for (int i = 0; i < num_act; i++)
					{
						x = layers[i].forward(x);
					}

					return x;
				}
			}
		}

		private class ViT_bigG_Clip : Module<Tensor, int, bool, bool, Tensor>
		{
			private readonly int adm_in_channels;

			private readonly Embedding token_embedding;
			private readonly Parameter positional_embedding;
			private readonly Transformer transformer;
			private readonly LayerNorm ln_final;
			private readonly Parameter text_projection;

			public ViT_bigG_Clip(long n_vocab = 49408, long n_token = 77, long num_layers = 32, long n_heads = 20, long embed_dim = 1280, long intermediate_size = 1280 * 4, Device? device = null, ScalarType? dtype = null) : base(nameof(ViT_bigG_Clip))
			{
				token_embedding = Embedding(n_vocab, embed_dim, device: device, dtype: dtype);
				positional_embedding = Parameter(zeros(size: [n_token, embed_dim], device: device, dtype: dtype));
				text_projection = Parameter(zeros(size: [embed_dim, embed_dim], device: device, dtype: dtype));
				transformer = new Transformer(num_layers, embed_dim, n_heads, intermediate_size, Activations.GELU, device: device, dtype: dtype);
				ln_final = LayerNorm(embed_dim, device: device, dtype: dtype);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x, int num_skip, bool with_final_ln, bool return_pooled)
			{
				using (NewDisposeScope())
				{
					Tensor input_ids = x;
					x = token_embedding.forward(x) + positional_embedding;
					x = transformer.forward(x, num_skip);
					if (with_final_ln || return_pooled)
					{
						x = ln_final.forward(x);
					}
					if (return_pooled)
					{
						x = x[torch.arange(x.shape[0], device: x.device), input_ids.to(type: ScalarType.Int32, device: x.device).argmax(dim: -1)];
						x = functional.linear(x, text_projection.transpose(0, 1));
					}
					return x.MoveToOuterDisposeScope();
				}
			}

			private class Transformer : Module<Tensor, int, Tensor>
			{
				private readonly ModuleList<ResidualAttentionBlock> resblocks;
				public Transformer(long num_layers, long embed_dim, long heads, long intermediate_size, Activations intermediate_activation, Device? device = null, ScalarType? dtype = null) : base(nameof(Transformer))
				{
					resblocks = new ModuleList<ResidualAttentionBlock>();
					for (int i = 0; i < num_layers; i++)
					{
						resblocks.append(new ResidualAttentionBlock(heads, embed_dim, intermediate_size, intermediate_activation, device: device, dtype: dtype));
					}
					RegisterComponents();
				}

				public override Tensor forward(Tensor x, int num_skip)
				{
					int num_act = num_skip > 0 ? resblocks.Count - num_skip : resblocks.Count;
					for (int i = 0; i < num_act; i++)
					{
						x = resblocks[i].forward(x);
					}
					return x;
				}
			}

			private class ResidualAttentionBlock : Module<Tensor, Tensor>
			{
				private readonly LayerNorm ln_1;
				private readonly LayerNorm ln_2;
				private readonly MultiheadAttention attn;
				private readonly Mlp mlp;

				public ResidualAttentionBlock(long n_head, long embed_dim, long intermediate_size, Activations activations = Activations.QuickGELU, Device? device = null, ScalarType? dtype = null) : base(nameof(ResidualAttentionBlock))
				{
					ln_1 = LayerNorm(embed_dim, device: device, dtype: dtype);
					attn = new MultiheadAttention(embed_dim, n_head, device: device, dtype: dtype);
					ln_2 = LayerNorm(embed_dim, device: device, dtype: dtype);
					mlp = new Mlp(embed_dim, intermediate_size, embed_dim, activations, device: device, dtype: dtype);
					RegisterComponents();
				}

				public override Tensor forward(Tensor x)
				{
					x += attn.forward(ln_1.forward(x));
					x += mlp.forward(ln_2.forward(x));
					return x;
				}
			}

			private class Mlp : Module<Tensor, Tensor>
			{
				private readonly Linear c_fc;
				private readonly Linear c_proj;
				private readonly Activations act_layer;
				public Mlp(long in_features, long? hidden_features = null, long? out_features = null, Activations act_layer = Activations.QuickGELU, bool bias = true, Device? device = null, ScalarType? dtype = null) : base(nameof(Mlp))
				{
					out_features ??= in_features;
					hidden_features ??= out_features;

					c_fc = Linear(in_features, (long)hidden_features, hasBias: bias, device: device, dtype: dtype);
					c_proj = Linear((long)hidden_features, (long)out_features, hasBias: bias, device: device, dtype: dtype);
					this.act_layer = act_layer;
					RegisterComponents();
				}

				public override Tensor forward(Tensor x)
				{
					x = c_fc.forward(x);

					switch (act_layer)
					{
						case Activations.ReLU:
							x = functional.relu(x);
							break;
						case Activations.SiLU:
							x = functional.silu(x);
							break;
						case Activations.QuickGELU:
							x = x * sigmoid(1.702 * x);
							break;
						case Activations.GELU:
							x = functional.gelu(x);
							break;
					}
					x = c_proj.forward(x);
					return x;
				}
			}

			private class MultiheadAttention : Module<Tensor, Tensor>
			{
				private readonly long heads;
				private readonly Parameter in_proj_weight;
				private readonly Parameter in_proj_bias;
				private readonly Linear out_proj;

				public MultiheadAttention(long embed_dim, long heads, Device? device = null, ScalarType? dtype = null) : base(nameof(MultiheadAttention))
				{
					this.heads = heads;
					in_proj_weight = Parameter(zeros([3 * embed_dim, embed_dim], device: device, dtype: dtype));
					in_proj_bias = Parameter(zeros([3 * embed_dim], device: device, dtype: dtype));
					out_proj = Linear(embed_dim, embed_dim, hasBias: true, device: device, dtype: dtype);

					RegisterComponents();
				}

				public override Tensor forward(Tensor x)
				{
					using (var _ = NewDisposeScope())
					{
						Tensor[] qkv = functional.linear(x, in_proj_weight, in_proj_bias).chunk(3, 2);
						Tensor q = qkv[0];
						Tensor k = qkv[1];
						Tensor v = qkv[2];
						Tensor output = attention(q, k, v, heads);
						//TensorInfo output = self_atten(to_q, to_k, to_v, this.heads);
						return out_proj.forward(output).MoveToOuterDisposeScope();
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

					var weight = matmul(q, k.transpose(-1, -2));
					var mask = ones_like(weight).triu(1).to(@bool);
					weight.masked_fill_(mask, float.NegativeInfinity);

					weight = weight / (float)Math.Sqrt(d_head);
					weight = functional.softmax(weight, dim: -1);

					var output = matmul(weight, v);
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
					Tensor output = functional.scaled_dot_product_attention(q, k, v, is_casual: true);
					output = output.transpose(1, 2);
					output = output.view(b, -1, heads * dim_head);
					return output;
				}
			}
		}

		internal class SDCliper : Module<Tensor, long, (Tensor, Tensor)>
		{
			private readonly ViT_L_Clip cond_stage_model;
			private readonly long n_token;
			private readonly long endToken;

			public SDCliper(long n_vocab = 49408, long n_token = 77, long num_layers = 12, long n_heads = 12, long embed_dim = 768, long intermediate_size = 768 * 4, long endToken = 49407, Device? device = null, ScalarType? dtype = null) : base(nameof(SDCliper))
			{
				this.n_token = n_token;
				this.endToken = endToken;
				cond_stage_model = new ViT_L_Clip(n_vocab, n_token, num_layers, n_heads, embed_dim, intermediate_size, device: device, dtype: dtype);
				RegisterComponents();
			}
			public override (Tensor, Tensor) forward(Tensor token, long num_skip)
			{
				using (NewDisposeScope())
				{
					Device device = cond_stage_model.parameters().First().device;
					long padLength = n_token - token.shape[1];
					Tensor token1 = functional.pad(token, [0, padLength, 0, 0], value: endToken);
					return (cond_stage_model.forward(token1, num_skip, true).MoveToOuterDisposeScope(), zeros(1).MoveToOuterDisposeScope());
				}
			}
		}

		internal class SDXLCliper : Module<Tensor, long, (Tensor, Tensor)>
		{
			private readonly Embedders conditioner;
			public SDXLCliper(long n_vocab = 49408, long n_token = 77, Device? device = null, ScalarType? dtype = null) : base(nameof(SDXLCliper))
			{
				conditioner = new Embedders(n_token, device: device, dtype: dtype);
				RegisterComponents();
			}

			public override (Tensor, Tensor) forward(Tensor token, long num_skip)
			{
				Device device = conditioner.parameters().First().device;
				token = token.to(device);
				return conditioner.forward(token);
			}

			private class Embedders : Module<Tensor, (Tensor, Tensor)>
			{
				private readonly ModuleList<Module> embedders;
				private readonly long n_token;
				private readonly long endToken;
				public Embedders(long n_token = 77, int endToken = 49407, Device? device = null, ScalarType? dtype = null) : base(nameof(Embedders))
				{
					this.n_token = n_token;
					this.endToken = endToken;
					Model model = new Model(device: device, dtype: dtype);
					embedders = ModuleList(new ViT_L_Clip(device: device, dtype: dtype), model);
					RegisterComponents();
				}
				public override (Tensor, Tensor) forward(Tensor token)
				{
					using (NewDisposeScope())
					{
						long padLength = n_token - token.shape[1];
						Tensor token1 = functional.pad(token, [0, padLength, 0, 0], value: endToken);
						Tensor token2 = functional.pad(token, [0, padLength, 0, 0]);

						Tensor vit_l_result = ((ViT_L_Clip)embedders[0]).forward(token1, 1, false);
						Tensor vit_bigG_result = ((Model)embedders[1]).forward(token2, 1, false, false);
						Tensor vit_bigG_vec = ((Model)embedders[1]).forward(token2, 0, false, true);
						Tensor crossattn = cat([vit_l_result, vit_bigG_result], -1);
						return (crossattn.MoveToOuterDisposeScope(), vit_bigG_vec.MoveToOuterDisposeScope());
					}
				}
			}

			private class Model : Module<Tensor, int, bool, bool, Tensor>
			{
				private readonly ViT_bigG_Clip model;
				public Model(Device? device = null, ScalarType? dtype = null) : base(nameof(Model))
				{
					model = new ViT_bigG_Clip(device: device, dtype: dtype);
					RegisterComponents();
				}
				public override Tensor forward(Tensor token, int num_skip, bool with_final_ln, bool return_pooled)
				{
					return model.forward(token, num_skip, with_final_ln, return_pooled);
				}
			}
		}

	}
}
