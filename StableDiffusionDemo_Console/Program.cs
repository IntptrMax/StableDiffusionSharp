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

			string i2iPrompt = "High quality, best quality, moon, grass, tree, boat.";

			SDDeviceType deviceType = SDDeviceType.CUDA;
			SDScalarType scalarType = SDScalarType.Float16;
			SDSamplerType samplerType = SDSamplerType.EulerAncestral;
			int step = 20;
			float cfg = 7.0f;
			long seed = 0;
			long img2imgSubSeed = 0;
			int width = 512;
			int height = 512;
			float strength = 0.75f;


			StableDiffusion sd = new StableDiffusion(deviceType, scalarType);
			Console.WriteLine("Loading model......");
			sd.LoadModel(modelPath);
			Console.WriteLine("Model loaded.");

			ImageMagick.MagickImage t2iImage = sd.TextToImage(prompt, nprompt, width, height, step, seed, cfg, samplerType);
			t2iImage.Write("output_t2i.png");

			ImageMagick.MagickImage i2iImage = sd.ImageToImage(t2iImage, i2iPrompt, nprompt, step, strength, seed, img2imgSubSeed, cfg, samplerType);
			i2iImage.Write("output_i2i.png");

			Console.WriteLine(@"Done. Images have been saved.");
		}
	}
}
