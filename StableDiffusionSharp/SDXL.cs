﻿using StableDiffusionSharp.Sampler;
using System.Diagnostics;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace StableDiffusionSharp
{
	public class SDXL : IDisposable
	{
		// Default parameters
		private float linear_start = 0.00085f;
		private float linear_end = 0.0120f;
		private float scale_factor = 0.13025f;
		private int num_timesteps_cond = 1;
		private int timesteps = 1000;

		// UNet config
		private int in_channels = 4;
		private int model_channels = 320;
		private int context_dim = 2048;
		private int num_head = 20;
		private float dropout = 0.0f;
		private int adm_in_channels = 2816;

		// first stage config:
		private int embed_dim = 4;
		private bool double_z = true;
		private int z_channels = 4;

		public class StepEventArgs : EventArgs
		{
			public int CurrentStep { get; }
			public int TotalSteps { get; }

			public StepEventArgs(int currentStep, int totalSteps)
			{
				CurrentStep = currentStep;
				TotalSteps = totalSteps;
			}
		}

		public event EventHandler<StepEventArgs> StepProgress;
		protected virtual void OnStepProgress(int currentStep, int totalSteps)
		{
			StepProgress?.Invoke(this, new StepEventArgs(currentStep, totalSteps));
		}

		private readonly Clip.SDXLCliper cliper;
		private readonly SDXLUnet diffusion;
		private readonly VAE.Decoder decoder;
		private readonly VAE.Encoder encoder;
		private Tokenizer tokenizer;

		private readonly Device device;
		private readonly ScalarType dtype;

		private int tempPromptHash;
		private Tensor tempTextContext;
		private Tensor tempVector;

		bool is_loaded = false;

		private static Tensor GetTimeEmbedding(Tensor timestep, int max_period = 10000, int dim = 320, bool repeat_only = false)
		{
			if (repeat_only)
			{
				return torch.repeat_interleave(timestep, dim);
			}
			else
			{
				int half = dim / 2;
				var freqs = torch.pow(max_period, -torch.arange(0, half, dtype: torch.float32) / half);
				var x = timestep * freqs.unsqueeze(0);
				x = torch.cat([x, x]);
				return torch.cat([torch.cos(x), torch.sin(x)], dim: -1);
			}
		}

		public SDXL(SDDeviceType deviceType = SDDeviceType.CUDA, SDScalarType scalarType = SDScalarType.Float16)
		{
			this.device = new Device((DeviceType)deviceType);
			this.dtype = (ScalarType)scalarType;
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			cliper = new Clip.SDXLCliper().to(float32).to(CPU);
			cliper.eval();
			diffusion = new SDXLUnet(model_channels, in_channels, num_head, context_dim, adm_in_channels, dropout).to(dtype).to(device);
			diffusion.eval();
			decoder = new VAE.Decoder(embed_dim: embed_dim, z_channels: z_channels).to(float32).to(CPU);
			decoder.eval();
			encoder = new VAE.Encoder(embed_dim: embed_dim, z_channels: z_channels, double_z: double_z).to(float32).to(CPU);
			encoder.eval();
		}

		public void LoadModel(string modelPath, string vocabPath = @".\models\clip\vocab.json", string mergesPath = @".\models\clip\merges.txt")
		{
			Dictionary<string, Tensor> state_dict = Path.GetExtension(modelPath).ToLower() switch
			{
				".safetensors" => ModelLoader.SafetensorsLoader.Load(modelPath),
				".pickle" => ModelLoader.PickleLoader.Load(modelPath),
				_ => throw new ArgumentException("Unknown model file extension")
			};

			var (diffusion_missing, diffusion_error) = diffusion.load_state_dict(state_dict, strict: false);
			var (cliper_missing, cliper_error) = cliper.load_state_dict(state_dict, strict: false);
			var (decoder_missing, decoder_error) = decoder.load_state_dict(state_dict, strict: false);
			var (encoder_missing, encoder_error) = encoder.load_state_dict(state_dict, strict: false);

			if (cliper_missing.Count + diffusion_missing.Count + decoder_missing.Count + encoder_missing.Count > 0)
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
				foreach (var key in encoder_missing)
				{
					Console.WriteLine(key);
				}
			}
			state_dict.Clear();
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

		private (Tensor, Tensor) Clip(string prompt, string nprompt)
		{
			CheckModelLoaded();
			if (tempPromptHash == (prompt + nprompt).GetHashCode())
			{
				return (this.tempTextContext, this.tempVector);
			}
			else
			{
				using (torch.no_grad())
				using (NewDisposeScope())
				{
					Tensor cond_tokens = tokenizer.Tokenize(prompt);
					(Tensor cond_context, Tensor cond_vector) = cliper.forward(cond_tokens);

					Tensor uncond_tokens = tokenizer.Tokenize(nprompt);
					(Tensor uncond_context, Tensor uncond_vector) = cliper.forward(uncond_tokens);

					Tensor context = torch.cat([cond_context, uncond_context]);
					this.tempPromptHash = (prompt + nprompt).GetHashCode();
					this.tempTextContext = context;
					this.tempVector = torch.cat([cond_vector, uncond_vector]);
					return (context.MoveToOuterDisposeScope(), tempVector.MoveToOuterDisposeScope());
				}
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
		public ImageMagick.MagickImage TextToImage(string prompt, string nprompt = "", int width = 512, int height = 512, int steps = 20, long seed = 0, float cfg = 7.0f, SDSamplerType samplerType = SDSamplerType.Euler)
		{
			CheckModelLoaded();


			if (steps < 1)
			{
				throw new ArgumentException("steps must be greater than 0");
			}
			if (cfg < 0.1)
			{
				throw new ArgumentException("cfg must be greater than 0.1");
			}
			if (cfg < 0.5)
			{
				throw new ArgumentException("cfg is too small, it may cause the model to be unstable");
			}

			seed = seed == 0 ? Random.Shared.NextInt64() : seed;
			Generator generator = torch.manual_seed(seed);
			torch.set_rng_state(generator.get_state());

			width = (width / 64) * 8;  // must be multiples of 64
			height = (height / 64) * 8; // must be multiples of 64
			Console.WriteLine("Device:" + device);
			Console.WriteLine("Type:" + dtype);
			Console.WriteLine("CFG:" + cfg);
			Console.WriteLine("Seed:" + seed);
			Console.WriteLine("Width:" + width * 8);
			Console.WriteLine("Height:" + height * 8);

			Stopwatch sp = Stopwatch.StartNew();
			Console.WriteLine("Clip is doing......");

			using (torch.no_grad())
			{
				(Tensor crossattn, Tensor vector) = Clip(prompt, nprompt);

				Console.WriteLine("Getting latents......");
				Tensor latents = torch.randn([1, 4, height, width]).to(dtype, device);

				BasicSampler sampler = samplerType switch
				{
					SDSamplerType.Euler => new EulerSampler(timesteps, linear_start, linear_end, num_timesteps_cond),
					SDSamplerType.EulerAncestral => new EulerAncestralSampler(timesteps, linear_start, linear_end, num_timesteps_cond),
					_ => throw new ArgumentException("Unknown sampler type")
				};

				sampler.SetTimesteps(steps);
				latents *= sampler.InitNoiseSigma();

				Console.WriteLine($"begin sampling");
				for (int i = 0; i < steps; i++)
				{
					Console.WriteLine($"steps:" + (i + 1));
					Tensor timestep = sampler.Timesteps[i];
					Tensor time_embedding = GetTimeEmbedding(timestep).to(dtype, device);
					Tensor input_latents = sampler.ScaleModelInput(latents, i);
					input_latents = input_latents.repeat(2, 1, 1, 1);
					Tensor x_out = diffusion.forward(input_latents, crossattn, time_embedding, vector);

					Tensor[] ret = x_out.chunk(2);
					Tensor output_cond = ret[0];
					Tensor output_uncond = ret[1];
					Tensor noisePred = cfg * (output_cond - output_uncond) + output_uncond;
					latents = sampler.Step(noisePred, i, latents, seed);
					//OnStepProgress(i + 1, steps);
				}

				Console.WriteLine($"end sampling");
				Console.WriteLine($"begin decoder");
				latents = latents / scale_factor;
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
				stringBuilder.AppendLine($"Steps: {steps}, CFG scale_factor: {cfg}, Seed: {seed}, Size: {width}x{height}, Version: StableDiffusionSharp");
				img.SetAttribute("parameters", stringBuilder.ToString());
				return img;
			}
		}


		public ImageMagick.MagickImage ImageToImage(ImageMagick.MagickImage orgImage, string prompt, string nprompt = "", int steps = 20, float strength = 0.75f, long seed = 0, long subSeed = 0, float cfg = 7.0f, SDSamplerType samplerType = SDSamplerType.Euler)
		{
			CheckModelLoaded();

			using (torch.no_grad())
			{
				Stopwatch sp = Stopwatch.StartNew();
				seed = seed == 0 ? Random.Shared.NextInt64() : seed;
				Generator generator = torch.manual_seed(seed);
				torch.set_rng_state(generator.get_state());

				Console.WriteLine("Clip is doing......");
				(Tensor context, Tensor pooled) = Clip(prompt, nprompt);

				Console.WriteLine("Getting latents......");
				Tensor inputTensor = Tools.GetTensorFromImage(orgImage).unsqueeze(0);
				inputTensor = inputTensor.to(dtype, device);
				inputTensor = inputTensor / 255.0f * 2 - 1.0f;
				Tensor lt = encoder.forward(inputTensor);

				Tensor[] mean_var = lt.chunk(2, 1);
				Tensor mean = mean_var[0];
				Tensor logvar = mean_var[1].clamp(-30, 20);
				Tensor std = torch.exp(0.5f * logvar);
				Tensor latents = mean + std * torch.randn_like(mean);

				latents = latents * scale_factor;
				int t_enc = (int)(strength * steps) - 1;

				BasicSampler sampler = samplerType switch
				{
					SDSamplerType.Euler => new EulerSampler(timesteps, linear_start, linear_end, num_timesteps_cond),
					SDSamplerType.EulerAncestral => new EulerAncestralSampler(timesteps, linear_start, linear_end, num_timesteps_cond),
					_ => throw new ArgumentException("Unknown sampler type")
				};

				sampler.SetTimesteps(steps);
				Tensor sigma_sched = sampler.Sigmas[(steps - t_enc - 1)..];
				Tensor noise = randn_like(latents);
				latents = latents + noise * sigma_sched.max();

				Console.WriteLine($"begin sampling");
				for (int i = 0; i < sigma_sched.NumberOfElements - 1; i++)
				{
					int index = steps - t_enc + i - 1;
					Console.WriteLine($"steps:" + (i + 1));
					Tensor timestep = sampler.Timesteps[index];
					Tensor time_embedding = GetTimeEmbedding(timestep).to(dtype, device);
					Tensor input_latents = sampler.ScaleModelInput(latents, index);
					input_latents = input_latents.repeat(2, 1, 1, 1);
					Tensor output = diffusion.forward(input_latents, context, time_embedding, torch.zeros(0));
					Tensor[] ret = output.chunk(2);
					Tensor output_cond = ret[0];
					Tensor output_uncond = ret[1];
					Tensor noisePred = cfg * (output_cond - output_uncond) + output_uncond;
					latents = sampler.Step(noisePred, index, latents, seed);
					//OnStepProgress(i + 1, steps);
				}
				Console.WriteLine($"end sampling");
				Console.WriteLine($"begin decoder");
				//latents = latents / scale_factor;
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
				stringBuilder.AppendLine($"Steps: {steps}, CFG scale_factor: {cfg}, Seed: {seed}, Size: {img.Width}x{img.Height}, Version: StableDiffusionSharp");
				img.SetAttribute("parameters", stringBuilder.ToString());
				return img;
			}
		}

		public void Dispose()
		{
			cliper?.Dispose();
			diffusion?.Dispose();
			decoder?.Dispose();
			encoder?.Dispose();
			tempTextContext?.Dispose();
		}

	}

}

