using StableDiffusionSharp;

namespace StableDiffusionDemo_Console
{
	internal class Program
	{
		static void Main(string[] args)
		{
			string modelPath = @".\sunshinemix.safetensors";
			string prompt = "High quality, best quality, sunset on sea, beach, tree.";
			string nprompt = "Bad quality, worst quality.";
			SDDeviceType deviceType = SDDeviceType.CUDA;
			SDScalarType scalarType = SDScalarType.Float16;
			int step = 20;
			float cfg = 7.0f;
			ulong seed = 0;
			int width = 512;
			int height = 512;

			StableDiffusion sd = new StableDiffusion(deviceType, scalarType);

			Console.WriteLine("Loading model......");
			sd.LoadModel(modelPath);
			Console.WriteLine("Model loaded.");

			ImageMagick.MagickImage image = sd.TextToImage(prompt, nprompt, width, height, step, seed, cfg);
			image.Write("output.png");
			Console.WriteLine(@"Done. Image is saved to .\output.png");
		}
	}
}
