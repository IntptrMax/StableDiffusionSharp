using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace StableDiffusionSharp
{
	public class Clip
	{
		private class CLIPEmbedding : Module<Tensor, Tensor>
		{
			internal readonly Embedding token_embedding;
			internal readonly Embedding position_embedding;
			internal CLIPEmbedding(long n_vocab, long n_embd, long n_token) : base(nameof(CLIPEmbedding))
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
			internal readonly LayerNorm layer_norm1;
			internal readonly LayerNorm layer_norm2;
			internal readonly CLIPAttention self_attn;
			internal readonly Mlp mlp;

			internal CLIPLayer(long n_head, long embed_dim, long intermediate_size, Activations activations = Activations.QuickGELU) : base(nameof(CLIPLayer))
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

		private enum Activations
		{
			ReLU,
			SiLU,
			QuickGELU,
			GELU
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

		public class Cliper : Module<Tensor, Tensor>
		{
			internal Sequential cond_stage_model;
			public Cliper(long n_vocab = 49408, long n_token = 77, long n_heads = 12, long embed_dim = 768, long intermediate_size = 768 * 4) : base(nameof(Cliper))
			{
				cond_stage_model = Sequential(("transformer", Sequential(("text_model", Sequential(
					[("embeddings", new CLIPEmbedding(n_vocab, embed_dim, n_token)),
					("encoder", new ClipEncoder(n_heads, embed_dim, n_heads, intermediate_size, Activations.GELU)),
					("final_layer_norm", LayerNorm(embed_dim))])))));
				RegisterComponents();
			}
			public override Tensor forward(Tensor token)
			{
				return cond_stage_model.forward(token);
			}
		}

	}
}
