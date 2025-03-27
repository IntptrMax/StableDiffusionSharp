using StableDiffusionSharp;

namespace StableDiffusionDemo_Console
{
	internal class Program
	{
		static void Main(string[] args)
		{
			string modelPath = @".\tPonynai3_v4.safetensors";
			//modelPath = @"Chilloutmix.safetensors";
			string esrganModelPath = @".\RealESRGAN_x4plus.pth";
			string prompt = "High quality, best quality, realistic, beach, trees, sunset on sea.";
			string nprompt = "2D, 3D, cartoon, painting, bad quality, worst quality.";
			string i2iPrompt = "High quality, best quality, moon, grass, tree, boat.";
			prompt = "cat with blue eyes";

			SDDeviceType deviceType = SDDeviceType.CUDA;
			SDScalarType scalarType = SDScalarType.Float16;
			SDSamplerType samplerType = SDSamplerType.Euler;
			int step = 20;
			float cfg = 7.0f;
			long seed = 0;
			long img2imgSubSeed = 0;
			int width = 512;
			int height = 512;
			float strength = 0.75f;

			// SDXL is not supported now.
			SDXL sdxl = new SDXL(deviceType, scalarType);
			sdxl.LoadModel(modelPath);
			ImageMagick.MagickImage sdxlT2Image = sdxl.TextToImage(prompt, nprompt, width, height, step, seed, cfg, samplerType);
			sdxlT2Image.Write("output_sdxl_t2i.png");

			StableDiffusion sd = new StableDiffusion(deviceType, scalarType);
			Console.WriteLine("Loading model......");
			sd.LoadModel(modelPath);
			Console.WriteLine("Model loaded.");

			ImageMagick.MagickImage t2iImage = sd.TextToImage(prompt, nprompt, width, height, step, seed, cfg, samplerType);
			t2iImage.Write("output_t2i.png");

			ImageMagick.MagickImage i2iImage = sd.ImageToImage(t2iImage, i2iPrompt, nprompt, step, strength, seed, img2imgSubSeed, cfg, samplerType);
			i2iImage.Write("output_i2i.png");

			sd.Dispose();
			GC.Collect();

			Console.WriteLine("Doing upscale......");
			Esrgan esrgan = new Esrgan(deviceType: deviceType, scalarType: scalarType);
			esrgan.LoadModel(esrganModelPath);
			ImageMagick.MagickImage upscaleImg = esrgan.UpScale(t2iImage);
			upscaleImg.Write("upscale.png");

			Console.WriteLine(@"Done. Images have been saved.");
		}
	}
}
