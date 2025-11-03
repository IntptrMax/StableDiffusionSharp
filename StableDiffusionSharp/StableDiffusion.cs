using StableDiffusionSharp.Modules;
using TorchSharp;
using static TorchSharp.torch;

namespace StableDiffusionSharp
{
	public class StableDiffusion : nn.Module
	{
		private SDModel model;
		private readonly Device device;
		private readonly ScalarType dtype;

		public class StepEventArgs : EventArgs
		{
			public int CurrentStep { get; }
			public int TotalSteps { get; }

			public ImageMagick.MagickImage VaeApproxImg { get; }

			public StepEventArgs(int currentStep, int totalSteps, ImageMagick.MagickImage vaeApproxImg)
			{
				CurrentStep = currentStep;
				TotalSteps = totalSteps;
				VaeApproxImg = vaeApproxImg;
			}
		}

		public event EventHandler<StepEventArgs> StepProgress;
		protected void OnStepProgress(int currentStep, int totalSteps, ImageMagick.MagickImage vaeApproxImg)
		{
			StepProgress?.Invoke(this, new StepEventArgs(currentStep, totalSteps, vaeApproxImg));
		}

		public StableDiffusion(SDDeviceType deviceType, SDScalarType scaleType) : base(nameof(StableDiffusion))
		{
			this.device = new Device((DeviceType)deviceType);
			this.dtype = (ScalarType)scaleType;
		}

		public void LoadModel(string modelPath, string vaeModelPath = "", string vocabPath = @".\models\clip\vocab.json", string mergesPath = @".\models\clip\merges.txt")
		{
			using (no_grad())
			{
				ModelType modelType = ModelLoader.ModelLoader.GetModelType(modelPath);
				Console.WriteLine($"Maybe you are using: {modelType}");
				model = modelType switch
				{
					ModelType.SD1 => new SD1(device: this.device, dtype: this.dtype),
					ModelType.SDXL => new SDXL(device: this.device, dtype: this.dtype),
					_ => throw new ArgumentException("Invalid model type")
				};
				model.LoadModel(modelPath, vaeModelPath, vocabPath, mergesPath);
				model.StepProgress += Model_StepProgress;
			}
		}

		private void Model_StepProgress(object? sender, SDModel.StepEventArgs e)
		{
			OnStepProgress(e.CurrentStep, e.TotalSteps, e.VAEApproxImg);
		}

		public ImageMagick.MagickImage TextToImage(string prompt, string nprompt = "", long clip_skip = 0, int width = 512, int height = 512, int steps = 20, long seed = 0, float cfg = 7.0f, SDSamplerType samplerType = SDSamplerType.Euler)
		{
			using (no_grad())
			{
				return model.TextToImage(prompt, nprompt, clip_skip, width, height, steps, seed, cfg, samplerType);
			}
		}

		public ImageMagick.MagickImage ImageToImage(ImageMagick.MagickImage orgImage, string prompt, string nprompt = "", long clip_skip = 0, int steps = 20, float strength = 0.75f, long seed = 0, long subSeed = 0, float cfg = 7.0f, SDSamplerType samplerType = SDSamplerType.Euler)
		{
			using (no_grad())
			{
				return model.ImageToImage(orgImage, prompt, nprompt, clip_skip, steps, strength, seed, subSeed, cfg, samplerType);
			}
		}

	}
}
