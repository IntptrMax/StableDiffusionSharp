using TorchSharp;
using static TorchSharp.torch;

namespace StableDiffusionSharp.ModelLoader
{
	internal static class ModelLoader
	{
		public static nn.Module LoadModel(this torch.nn.Module module, string fileName, string maybeAddHeaderInBlock = "")
		{
			string extension = Path.GetExtension(fileName).ToLower();
			if (extension == ".pt" || extension == ".ckpt" || extension == ".pth")
			{
				PickleLoader pickleLoader = new PickleLoader();
				return pickleLoader.LoadPickle(module, fileName, maybeAddHeaderInBlock);
			}
			else if (extension == ".safetensors")
			{
				SafetensorsLoader safetensorsLoader = new SafetensorsLoader();
				return safetensorsLoader.LoadSafetensors(module, fileName, maybeAddHeaderInBlock);
			}
			else
			{
				throw new ArgumentException("Invalid file extension");
			}


		}
	}
}
