using ImageMagick;
using TorchSharp;
using static TorchSharp.torch;

namespace StableDiffusionSharp
{
	internal class Tools
	{
		internal static Tensor GetTensorFromImage(MagickImage image)
		{
			using (MemoryStream memoryStream = new MemoryStream())
			{
				image.Write(memoryStream, MagickFormat.Png);
				memoryStream.Position = 0;
				return torchvision.io.read_image(memoryStream);
			}
		}

		public static MagickImage GetImageFromTensor(Tensor tensor)
		{
			MemoryStream memoryStream = new MemoryStream();
			torchvision.io.write_png(tensor.cpu(), memoryStream);
			memoryStream.Position = 0;
			return new MagickImage(memoryStream, MagickFormat.Png);
		}

	}
}
