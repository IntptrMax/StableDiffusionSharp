﻿using StableDiffusionSharp.ModelLoader;
using StableDiffusionSharp.Sampler;
using System.Diagnostics;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace StableDiffusionSharp.Modules
{
	public class SDModel : IDisposable
	{
		// Default parameters
		private float linear_start = 0.00085f;
		private float linear_end = 0.0120f;
		private int num_timesteps_cond = 1;
		private int timesteps = 1000;
		internal float scale_factor = 0.18215f;
		internal int adm_in_channels = 2816;

		// UNet config
		internal int in_channels = 4;
		internal int model_channels = 320;
		internal int context_dim = 768;
		internal int num_head = 8;
		internal float dropout = 0.0f;

		// first stage config:
		internal int embed_dim = 4;
		internal bool double_z = true;
		internal int z_channels = 4;

		public class StepEventArgs : EventArgs
		{
			public int CurrentStep { get; }
			public int TotalSteps { get; }
			public ImageMagick.MagickImage VAEApproxImg { get; }

			public StepEventArgs(int currentStep, int totalSteps, ImageMagick.MagickImage vAEApproxImg)
			{
				CurrentStep = currentStep;
				TotalSteps = totalSteps;
				VAEApproxImg = vAEApproxImg;
			}
		}

		public event EventHandler<StepEventArgs> StepProgress;
		protected void OnStepProgress(int currentStep, int totalSteps, ImageMagick.MagickImage vaeApproxImg)
		{
			StepProgress?.Invoke(this, new StepEventArgs(currentStep, totalSteps, vaeApproxImg));
		}

		internal Module<Tensor, long, (Tensor, Tensor)> cliper;
		internal Module<Tensor, Tensor, Tensor, Tensor, Tensor> diffusion;
		private VAE.Decoder decoder;
		private VAE.Encoder encoder;
		private Tokenizer tokenizer;
		private VAEApprox vaeApprox;

		internal Device device;
		internal ScalarType dtype;

		private int tempPromptHash;
		private Tensor tempTextContext;
		private Tensor tempPooled;

		bool is_loaded = false;

		public SDModel(Device? device = null, ScalarType? dtype = null)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			this.device = device ?? torch.CPU;
			this.dtype = dtype ?? torch.float32;
		}

		public virtual void LoadModel(string modelPath, string vaeModelPath, string vocabPath = @".\models\clip\vocab.json", string mergesPath = @".\models\clip\merges.txt")
		{
			is_loaded = false;
			ModelType modelType = ModelLoader.ModelLoader.GetModelType(modelPath);

			cliper = modelType switch
			{
				ModelType.SD1 => new Clip.SDCliper(device: device, dtype: dtype),
				ModelType.SDXL => new Clip.SDXLCliper(device: device, dtype: dtype),
				_ => throw new ArgumentException("Invalid model type")
			};
			cliper.eval();

			diffusion = modelType switch
			{
				ModelType.SD1 => new SDUnet(model_channels, in_channels, num_head, context_dim, dropout, device: device, dtype: dtype),
				ModelType.SDXL => new SDXLUnet(model_channels, in_channels, num_head, context_dim, adm_in_channels, dropout, device: device, dtype: dtype),
				_ => throw new ArgumentException("Invalid model type")
			};
			diffusion.eval();

			decoder = new VAE.Decoder(embed_dim: embed_dim, z_channels: z_channels, device: device, dtype: dtype);
			decoder.eval();
			encoder = new VAE.Encoder(embed_dim: embed_dim, z_channels: z_channels, double_z: double_z, device: device, dtype: dtype);
			encoder.eval();

			vaeApprox = new VAEApprox(4, device, dtype);
			vaeApprox.eval();

			vaeModelPath = string.IsNullOrEmpty(vaeModelPath) ? modelPath : vaeModelPath;

			cliper.LoadModel(modelPath);
			diffusion.LoadModel(modelPath);
			decoder.LoadModel(vaeModelPath, "first_stage_model.");
			encoder.LoadModel(vaeModelPath, "first_stage_model.");

			string vaeApproxPath = modelType switch
			{
				ModelType.SD1 => @".\Models\VAEApprox\vaeapp_sd15.pth",
				ModelType.SDXL => @".\Models\VAEApprox\xlvaeapp.pth",
				_ => throw new ArgumentException("Invalid model type")
			};

			vaeApprox.LoadModel(vaeApproxPath);

			tokenizer = new Tokenizer(vocabPath, mergesPath);
			is_loaded = true;

			GC.Collect();
		}

		private void CheckModelLoaded()
		{
			if (!is_loaded)
			{
				throw new InvalidOperationException("Model not loaded");
			}
		}

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
				x = torch.cat(new Tensor[] { x, x });
				return torch.cat(new Tensor[] { torch.cos(x), torch.sin(x) }, dim: -1);
			}
		}

		private (Tensor, Tensor) Clip(string prompt, string nprompt, long clip_skip)
		{
			CheckModelLoaded();
			if (tempPromptHash != (prompt + nprompt).GetHashCode())
			{
				using (no_grad())
				using (NewDisposeScope())
				{
					Tensor cond_tokens = tokenizer.Tokenize(prompt).to(device);
					(Tensor cond_context, Tensor cond_pooled) = cliper.forward(cond_tokens, clip_skip);
					Tensor uncond_tokens = tokenizer.Tokenize(nprompt).to(device);
					(Tensor uncond_context, Tensor uncond_pooled) = cliper.forward(uncond_tokens, clip_skip);
					Tensor context = cat(new Tensor[] { cond_context, uncond_context });
					tempPromptHash = (prompt + nprompt).GetHashCode();
					tempTextContext = context;
					tempPooled = cat(new Tensor[] { cond_pooled, uncond_pooled });
					tempTextContext = tempTextContext.MoveToOuterDisposeScope();
					tempPooled = tempPooled.MoveToOuterDisposeScope();
				}
			}
			return (tempTextContext, tempPooled);
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
		public virtual ImageMagick.MagickImage TextToImage(string prompt, string nprompt = "", long clip_skip = 0, int width = 512, int height = 512, int steps = 20, long seed = 0, float cfg = 7.0f, SDSamplerType samplerType = SDSamplerType.Euler)
		{
			CheckModelLoaded();

			using (no_grad())
			{
				if (steps < 1)
				{
					throw new ArgumentException("steps must be greater than 0");
				}
				if (cfg < 0.5)
				{
					throw new ArgumentException("cfg is too small, it may cause the image to be too noisy");
				}

				seed = seed == 0 ? Random.Shared.NextInt64() : seed;
				set_rng_state(manual_seed(seed).get_state());

				width = width / 64 * 8;  // must be multiples of 64
				height = height / 64 * 8; // must be multiples of 64
				Console.WriteLine("Device:" + device);
				Console.WriteLine("Type:" + dtype);
				Console.WriteLine("CFG:" + cfg);
				Console.WriteLine("Seed:" + seed);
				Console.WriteLine("Width:" + width * 8);
				Console.WriteLine("Height:" + height * 8);

				Stopwatch sp = Stopwatch.StartNew();
				Console.WriteLine("Clip is doing......");
				(Tensor context, Tensor vector) = Clip(prompt, nprompt, clip_skip);
				using var _ = NewDisposeScope();
				Console.WriteLine("Getting latents......");
				Tensor latents = randn(new long[] { 1, 4, height, width }).to(dtype, device);

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
					Tensor approxTensor = vaeApprox.forward(latents);
					approxTensor = approxTensor * 127.5 + 127.5;
					approxTensor = approxTensor.clamp(0, 255).@byte().cpu();
					ImageMagick.MagickImage approxImg = Tools.GetImageFromTensor(approxTensor);
					OnStepProgress(i + 1, steps, approxImg);
					Tensor timestep = sampler.Timesteps[i];
					Tensor time_embedding = GetTimeEmbedding(timestep);
					Tensor input_latents = sampler.ScaleModelInput(latents, i);
					input_latents = input_latents.repeat(2, 1, 1, 1);
					Tensor output = diffusion.forward(input_latents, context, time_embedding, vector);
					Tensor[] ret = output.chunk(2);
					Tensor output_cond = ret[0];
					Tensor output_uncond = ret[1];
					output = cfg * (output_cond - output_uncond) + output_uncond;
					latents = sampler.Step(output, i, latents, seed);
				}
				Console.WriteLine($"end sampling");
				Console.WriteLine($"begin decoder");
				latents = latents / scale_factor;
				Tensor image = decoder.forward(latents);
				Console.WriteLine($"end decoder");


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
				sp.Stop();
				Console.WriteLine($"Total time is: {sp.ElapsedMilliseconds} ms.");
				return img;
			}
		}


		public virtual ImageMagick.MagickImage ImageToImage(ImageMagick.MagickImage orgImage, string prompt, string nprompt = "", long clip_skip = 0, int steps = 20, float strength = 0.75f, long seed = 0, long subSeed = 0, float cfg = 7.0f, SDSamplerType samplerType = SDSamplerType.Euler)
		{
			CheckModelLoaded();

			using (no_grad())
			{
				Stopwatch sp = Stopwatch.StartNew();
				seed = seed == 0 ? Random.Shared.NextInt64() : seed;
				Generator generator = manual_seed(seed);
				set_rng_state(generator.get_state());

				Console.WriteLine("Clip is doing......");
				(Tensor context, Tensor vector) = Clip(prompt, nprompt, clip_skip);

				Console.WriteLine("Getting latents......");
				Tensor inputTensor = Tools.GetTensorFromImage(orgImage).unsqueeze(0);
				inputTensor = inputTensor.to(dtype, device);
				inputTensor = inputTensor / 255.0f * 2 - 1.0f;
				Tensor lt = encoder.forward(inputTensor);

				Tensor[] mean_var = lt.chunk(2, 1);
				Tensor mean = mean_var[0];
				Tensor logvar = mean_var[1].clamp(-30, 20);
				Tensor std = exp(0.5f * logvar);
				Tensor latents = mean + std * randn_like(mean);

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
					Tensor approxTensor = vaeApprox.forward(latents);
					approxTensor = approxTensor * 127.5 + 127.5;
					approxTensor = approxTensor.clamp(0, 255).@byte().cpu();
					ImageMagick.MagickImage approxImg = Tools.GetImageFromTensor(approxTensor);
					OnStepProgress(i + 1, steps, approxImg);

					int index = steps - t_enc + i - 1;
					Tensor timestep = sampler.Timesteps[index];
					Tensor time_embedding = GetTimeEmbedding(timestep);
					Tensor input_latents = sampler.ScaleModelInput(latents, index);
					input_latents = input_latents.repeat(2, 1, 1, 1);
					Tensor output = diffusion.forward(input_latents, context, time_embedding, vector);
					Tensor[] ret = output.chunk(2);
					Tensor output_cond = ret[0];
					Tensor output_uncond = ret[1];
					Tensor noisePred = cfg * (output_cond - output_uncond) + output_uncond;
					latents = sampler.Step(noisePred, index, latents, seed);
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

