# StableDiffusionSharp

**Use Stable diffusion with C# only.**

StableDiffusionSharp is an image generating software. With the help of torchsharp, stable diffusion can run without python.

![无标题](https://github.com/user-attachments/assets/5c19e26e-52d3-45eb-aa15-ae27351dfabd)

## Features

- Written in C# only.
- Can load .safetensors or .ckpt model directly.
- Cuda support.
- Use SDPA for speed-up and save vram.
- Text2Image support.
- SD1.5 support.
- Image2Image support.
- Esrgan 4x support.
- Nuget package.

For SD1.5 Text to Image, it cost about 3G VRAM and 2.4 seconds for Generating a 512*512 image in 20 step.

## Work to do

- Lora support.
- Tiled VAE.
- SDXL support.

## How to use

You can download the code or add it from nuget.

    dotnet add package IntptrMax.YoloSharp

Or use the code directly.

Please add one of libtorch-cpu, libtorch-cuda-12.1, libtorch-cuda-12.1-win-x64 or libtorch-cuda-12.1-linux-x64 version 2.5.1.0 to execute.

You have to download sd1.5 model first.
If you want to use esrgan for upscaling, you have to download model from [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

Now you can use it like the code below.

    string modelPath = @".\sunshinemix.safetensors";
    string esrganModelPath = @".\RealESRGAN_x4plus.pth";
    string prompt = "High quality, best quality, sunset on sea, beach, tree.";
    string nprompt = "Bad quality, worst quality.";
    string i2iPrompt = "High quality, best quality, moon, grass, tree, boat.";

    StableDiffusion sd = new StableDiffusion();
    sd.LoadModel(modelPath);

    ImageMagick.MagickImage t2iImage = sd.TextToImage(prompt, nprompt);
    t2iImage.Write("output_t2i.png");

    ImageMagick.MagickImage i2iImage = sd.ImageToImage(t2iImage, i2iPrompt, nprompt);
    i2iImage.Write("output_i2i.png");

    Esrgan esrgan = new Esrgan();
    esrgan.LoadModel(esrganModelPath);
    ImageMagick.MagickImage upscaleImg = esrgan.UpScale(t2iImage);
    upscaleImg.Write("upscale.png");

    Console.WriteLine(@"Done. Images have been saved.");
