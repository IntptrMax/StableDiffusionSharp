# StableDiffusionSharp

**Use Stable diffusion with C# only.**

StableDiffusionSharp is an image generating software. With the help of torchsharp, stable diffusion can run without python.

![Demo](../Assets/Demo.jpg)

## Features

- Written in C# only.
- Can load .safetensors or .ckpt model directly.
- Cuda support.
- Use SDPA for speed-up and save vram in fp16.
- Text2Image support.
- Image2Image support.
- SD1.5 support.
- SDXL support.
- VAEApprox support.
- Esrgan 4x support.
- Nuget package support.

For SD1.5 Text to Image, it cost about 3G VRAM and 2.4 seconds for Generating a 512*512 image in 20 step.

## Work to do

- Lora support.
- ControlNet support.
- Inpaint support.
- Tiled VAE.

## How to use

You can download the code or add it from nuget.

    dotnet add package IntptrMax.YoloSharp

Or use the code directly.

> [!NOTE]Please add one of libtorch-cpu, libtorch-cuda-12.1, libtorch-cuda-12.1-win-x64 or libtorch-cuda-12.1-linux-x64 version 2.5.1.0 to execute.

You have to download sd1.5 model first.
If you want to use esrgan for upscaling, you have to download model from [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

Now you can use it like the code below.

    static void Main(string[] args)
    {
        string sdModelPath = @".\Chilloutmix.safetensors";
        string vaeModelPath = @".\vae.safetensors";

        string esrganModelPath = @".\RealESRGAN_x4plus.pth";
        string i2iPrompt = "High quality, best quality, moon, grass, tree, boat.";
        string prompt = "cat with blue eyes";
        string nprompt = "";

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
        long clipSkip = 2;

        StableDiffusion sd = new StableDiffusion(deviceType, scalarType);
        sd.StepProgress += Sd_StepProgress;
        Console.WriteLine("Loading model......");
        sd.LoadModel(sdModelPath, vaeModelPath);
        Console.WriteLine("Model loaded.");

        ImageMagick.MagickImage t2iImage = sd.TextToImage(prompt, nprompt, clipSkip, width, height, step, seed, cfg, samplerType);
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

    private static void Sd_StepProgress(object? sender, StableDiffusion.StepEventArgs e)
    {
        Console.WriteLine($"Progress: {e.CurrentStep}/{e.TotalSteps}");
    }
