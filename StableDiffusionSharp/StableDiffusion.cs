using System.Diagnostics;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace StableDiffusionSharp
{
	public class StableDiffusion
	{
		private Clip.Cliper cliper;
		private Diffusion diffusion;
		private VAE.Decoder decoder;
		private Device device;
		private ScalarType dtype;
		private Tokenizer tokenizer;

		private int promptHash = 0;
		private Tensor context;

		bool is_loaded = false;

		private static Tensor GetTimeEmbedding(float timestep)
		{
			var freqs = torch.pow(10000, -torch.arange(0, 160, dtype: torch.float32) / 160);
			var x = torch.tensor(new float[] { timestep }, dtype: torch.float32)[torch.TensorIndex.Colon, torch.TensorIndex.None] * freqs[torch.TensorIndex.None];
			return torch.cat(new Tensor[] { torch.cos(x), torch.sin(x) }, dim: -1);
		}

		public StableDiffusion(SDDeviceType deviceType = SDDeviceType.CUDA, SDScalarType scalarType = SDScalarType.Float16)
		{
			this.device = new Device((DeviceType)deviceType);
			this.dtype = (ScalarType)scalarType;
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			cliper = new Clip.Cliper();
			diffusion = new Diffusion(320, 4);
			decoder = new VAE.Decoder();
		}

		public void LoadModel(string modelPath, string vocabPath = @".\models\clip\vocab.json", string mergesPath = @".\models\clip\merges.txt")
		{
			Dictionary<string, Tensor> state_dict = Path.GetExtension(modelPath).ToLower() switch
			{
				".safetensors" => ModelLoader.SafetensorsLoader.Load(modelPath),
				".pickle" => ModelLoader.PickleLoader.Load(modelPath),
				_ => throw new ArgumentException("Unknown model file extension")
			};

			var (cliper_missing, cliper_error) = cliper.load_state_dict(state_dict, strict: false);
			cliper.to(device, dtype);
			cliper.eval();

			var (diffusion_missing, diffusion_error) = diffusion.load_state_dict(state_dict, strict: false);
			diffusion.to(device, dtype);
			diffusion.eval();

			var (decoder_missing, decoder_error) = decoder.load_state_dict(state_dict, strict: false);
			decoder.to(device, dtype);
			decoder.eval();

			if (cliper_missing.Count + diffusion_missing.Count + decoder_missing.Count > 0)
			{
				Console.WriteLine("Missing keys in model loading:");
				foreach (var key in cliper_missing)
				{
					Console.WriteLine(key);
				}
				foreach (var key in diffusion_missing)
				{
					Console.WriteLine(key);
				}
				foreach (var key in decoder_missing)
				{
					Console.WriteLine(key);
				}
			}

			tokenizer = new Tokenizer(vocabPath, mergesPath);
			is_loaded = true;
		}

		private void CheckModelLoaded()
		{
			if (!is_loaded)
			{
				throw new InvalidOperationException("Model not loaded");
			}
		}

		private void Clip(string prompt, string nprompt)
		{
			CheckModelLoaded();
			if (promptHash != (prompt + nprompt).GetHashCode())
			{
				Tensor cond_tokens = tokenizer.Tokenize(prompt).to(device);
				Tensor cond_context = cliper.forward(cond_tokens);
				Tensor uncond_tokens = tokenizer.Tokenize(nprompt).to(device);
				Tensor uncond_context = cliper.forward(uncond_tokens);
				Tensor context = torch.cat([cond_context, uncond_context]).to(dtype, device);
				this.promptHash = (prompt + nprompt).GetHashCode();
				this.context = context;
			}
		}

		/// <summary>
		/// Generate image from text
		/// </summary>
		/// <param name="prompt">Prompt</param>
		/// <param name="nprompt">Negtive Prompt</param>
		/// <param name="width">Image width, must be multiples of 64, otherwise, it will be resized</param>
		/// <param name="height">Image width, must be multiples of 64, otherwise, it will be resized</param>
		/// <param name="steps">Step to generate image</param>
		/// <param name="seed">Random seed for generating image, it will get random when the value is 0</param>
		/// <param name="cfg">Classifier Free Guidance</param>
		public ImageMagick.MagickImage TextToImage(string prompt, string nprompt, int width = 512, int height = 512, int steps = 20, ulong seed = 0, float cfg = 7.0f)
		{
			CheckModelLoaded();
			seed = seed == 0 ? (ulong)Random.Shared.Next(0, int.MaxValue) : seed;
			steps = steps == 0 ? 20 : steps;
			cfg = cfg == 0 ? 7.0f : cfg;

			width = (width / 64) * 8;
			height = (height / 64) * 8;

			Console.WriteLine("Device:" + device);
			Console.WriteLine("Type:" + dtype);
			Console.WriteLine("CFG:" + cfg);
			Console.WriteLine("Seed:" + seed);
			Console.WriteLine("Width:" + width * 8);
			Console.WriteLine("Height:" + height * 8);

			using (torch.no_grad())
			{
				Stopwatch sp = Stopwatch.StartNew();

				Console.WriteLine("Clip is doing......");
				Clip(prompt, nprompt);

				Console.WriteLine("Getting latents......");
				long[] noise_shape = new long[] { 1, 4, height, width };
				var latents = torch.randn(noise_shape, generator: new Generator(seed, device)).to(dtype, device);

				EulerDiscreteScheduler sampler = new EulerDiscreteScheduler();

				sampler.SetTimesteps(steps, device);
				latents *= sampler.InitNoiseSigma();
				Console.WriteLine($"begin steps");
				for (int i = 0; i < steps; i++)
				{
					Console.WriteLine($"steps:" + i);
					var timestep = sampler.timesteps_[i];
					var time_embedding = GetTimeEmbedding(timestep.ToSingle()).to(dtype, device);
					var input_latents = sampler.ScaleModelInput(latents, timestep);
					input_latents = input_latents.repeat(2, 1, 1, 1).to(dtype, device); ;
					var output = diffusion.forward(input_latents, context, time_embedding);
					var ret = output.chunk(2);
					var output_cond = ret[0];
					var output_uncond = ret[1];
					output = cfg * (output_cond - output_uncond) + output_uncond;
					latents = sampler.Step(output, timestep, latents);
				}
				Console.WriteLine($"end steps");
				Console.WriteLine($"begin decoder");
				Tensor image = decoder.forward(latents);
				Console.WriteLine($"end decoder");

				sp.Stop();
				Console.WriteLine($"Total time is: {sp.ElapsedMilliseconds} ms.");
				image = ((image + 0.5) * 255.0f).clamp(0, 255).@byte().cpu();

				ImageMagick.MagickImage img = Tools.GetImageFromTensor(image);

				StringBuilder stringBuilder = new StringBuilder();
				stringBuilder.AppendLine(prompt);
				if (!string.IsNullOrEmpty(nprompt))
				{
					stringBuilder.AppendLine("Negative prompt: " + nprompt);
				}
				stringBuilder.AppendLine($"Steps: {steps}, CFG scale: {cfg}, Seed: {seed}, Size: {width}x{height}, Version: StableDiffusionSharp");
				img.SetAttribute("parameters", stringBuilder.ToString());
				return img;
			}



		}


	}

}

