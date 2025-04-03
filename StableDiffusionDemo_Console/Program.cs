using StableDiffusionSharp;

namespace StableDiffusionDemo_Console
{
	internal class Program
	{

		static void Main(string[] args)
		{
			string sdModelPath = @".\realDream_sdxlPony15.safetensors";
			string vaeModelPath = @".\sdxl.vae.safetensors";

			string esrganModelPath = @".\RealESRGAN_x4plus.pth";
			string i2iPrompt = "High quality, best quality, moon, grass, tree, boat.";
			string prompt = "cat with blue eyes";
			string nprompt = "";

			SDDeviceType deviceType = SDDeviceType.CUDA;
			SDScalarType scalarType = SDScalarType.Float16;
			SDSamplerType samplerType = SDSamplerType.EulerAncestral;
			int step = 30;
			float cfg = 7.0f;
			long seed = 0;
			long img2imgSubSeed = 0;
			int width = 512;
			int height = 512;
			float strength = 0.75f;
			long clipSkip = 2;

			StableDiffusion sd = new StableDiffusion(deviceType, scalarType);
			Console.WriteLine("Loading model......");
			sd.LoadModel(sdModelPath, vaeModelPath);
			Console.WriteLine("Model loaded.");

			ImageMagick.MagickImage t2iImage = sd.TextToImage(prompt, nprompt, clipSkip, width, height, step, seed, cfg, samplerType);
			t2iImage.Write("output_t2i.png");

			t2iImage = sd.TextToImage(prompt, nprompt, clipSkip, width, height, step, seed, cfg, samplerType);
			t2iImage.Write("output_t2i.png");

			ImageMagick.MagickImage i2iImage = sd.ImageToImage(t2iImage, i2iPrompt, nprompt, clipSkip, step, strength, seed, img2imgSubSeed, cfg, samplerType);
			i2iImage.Write("output_i2i.png");

			sd.Dispose();
			GC.Collect();

			Console.WriteLine("Doing upscale......");
			StableDiffusionSharp.Modules.Esrgan esrgan = new StableDiffusionSharp.Modules.Esrgan(deviceType: deviceType, scalarType: scalarType);
			esrgan.LoadModel(esrganModelPath);
			ImageMagick.MagickImage upscaleImg = esrgan.UpScale(t2iImage);
			upscaleImg.Write("upscale.png");

			Console.WriteLine(@"Done. Images have been saved.");
		}
	}
}
