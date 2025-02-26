using StableDiffusionSharp;
using System.Diagnostics;
using TorchSharp;
using static TorchSharp.torch;

namespace StableDiffusionDemo_Console
{
	internal class Program
	{
		static void Main(string[] args)
		{
			Generate();
		}

		public static Tensor GetTimeEmbedding(float timestep)
		{
			var freqs = torch.pow(10000, -torch.arange(0, 160, dtype: torch.float32) / 160);
			var x = torch.tensor(new float[] { timestep }, dtype: torch.float32)[torch.TensorIndex.Colon, torch.TensorIndex.None] * freqs[torch.TensorIndex.None];
			return torch.cat(new Tensor[] { torch.cos(x), torch.sin(x) }, dim: -1);
		}

		public static void Generate()
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

			int num_inference_steps = 20;
			Device device = CUDA;
			float cfg = 4.0f;
			ulong seed = (ulong)new Random().Next(0, int.MaxValue);
			ScalarType dtype = ScalarType.Float16;
			int noise_height = 64; // Output image height is 8 times of this
			int noise_width = 64; // Output image width is 8 times of this
			string modelname = @".\sunshinemix_PrunedFp16.safetensors";

			Console.WriteLine("Device:" + device);
			Console.WriteLine("CFG:" + cfg);
			Console.WriteLine("Seed:" + seed);

			Dictionary<string, Tensor> state_dict = StableDiffusionSharp.ModelLoader.SafetensorsLoader.Load(modelname);

			Console.WriteLine("Loading clip......");
			Clip.Cliper cliper = new Clip.Cliper();
			var (cliper_missing, cliper_error) = cliper.load_state_dict(state_dict, strict: false);
			cliper.to(device, dtype);
			cliper.eval();

			Console.WriteLine("Loading tokenizer......");
			string VocabPath = @".\models\clip\vocab.json";
			string MergesPath = @".\models\clip\merges.txt";
			var tokenizer = new Tokenizer(VocabPath, MergesPath);

			Console.WriteLine("Loading UNet......");
			Diffusion diffusioner = new Diffusion(320, 4);
			var (unet_missing, unet_error) = diffusioner.load_state_dict(state_dict, strict: false);
			diffusioner.to(device, dtype);
			diffusioner.eval();

			Console.WriteLine("Loading Vae Decoder......");
			VAE.Decoder decoder = new VAE.Decoder();
			var (vae_missing, vae_error) = decoder.load_state_dict(state_dict, strict: false);
			decoder.to(device, dtype);
			decoder.eval();

			using (torch.no_grad())
			{
				Stopwatch sp = Stopwatch.StartNew();

				string prompt = "High quality, best quality, sunset on sea, beach, tree.";
				string uncond_prompts = "Bad quality, worst quality.";

				Console.WriteLine("Clip is doing......");
				var cond_tokens = tokenizer.Tokenize(prompt).to(device);
				var cond_context = cliper.forward(cond_tokens);
				var uncond_tokens = tokenizer.Tokenize(uncond_prompts).to(device);
				var uncond_context = cliper.forward(uncond_tokens);
				var context = torch.cat([cond_context, uncond_context]).to(dtype, device);

				Console.WriteLine("Getting latents......");
				long[] noise_shape = new long[] { 1, 4, noise_height, noise_width };
				Generator generator = new Generator(seed, device);
				var latents = torch.randn(noise_shape, generator: generator);
				latents = latents.to(dtype, device);
				var sampler = new EulerDiscreteScheduler();

				sampler.SetTimesteps(num_inference_steps, device);
				latents *= sampler.InitNoiseSigma();
				Console.WriteLine($"begin step");
				for (int i = 0; i < num_inference_steps; i++)
				{
					Console.WriteLine($"step:" + i);
					var timestep = sampler.timesteps_[i];
					var time_embedding = GetTimeEmbedding(timestep.ToSingle()).to(dtype, device);
					var input_latents = sampler.ScaleModelInput(latents, timestep);
					input_latents = input_latents.repeat(2, 1, 1, 1).to(dtype, device); ;
					var output = diffusioner.forward(input_latents, context, time_embedding);
					var ret = output.chunk(2);
					var output_cond = ret[0];
					var output_uncond = ret[1];
					output = cfg * (output_cond - output_uncond) + output_uncond;
					latents = sampler.Step(output, timestep, latents);
				}
				Console.WriteLine($"end step");
				Console.WriteLine($"begin decoder");
				var images = decoder.forward(latents);
				Console.WriteLine($"end decoder");

				sp.Stop();
				Console.WriteLine($"Total time is: {sp.ElapsedMilliseconds} ms.");
				images = ((images + 0.5) * 255.0f).clamp(0, 255).@byte().cpu();
				torchvision.io.write_jpeg(images, "result.jpg");
				Console.WriteLine(@"Image has been saved to .\result.jpg");
			}
		}
	}
}
