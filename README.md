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

For SD1.5 Text to Image, it cost about 3G VRAM and 2.4 seconds for Generating a 512*512 image in 20 step.

## Work to do 

- Image2Image.
- Lora support.
- Nuget package.
- Tiled VAE.
- SDXL support.
