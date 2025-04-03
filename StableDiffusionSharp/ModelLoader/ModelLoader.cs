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

		public static ModelType GetModelType(string ModelPath)
		{
			string extension = Path.GetExtension(ModelPath).ToLower();
			List<TensorInfo> tensorInfos = new List<TensorInfo>();

			if (extension == ".pt" || extension == ".ckpt" || extension == ".pth")
			{
				PickleLoader pickleLoader = new PickleLoader();
				tensorInfos = pickleLoader.ReadTensorsInfoFromFile(ModelPath);
			}
			else if (extension == ".safetensors")
			{
				SafetensorsLoader safetensorsLoader = new SafetensorsLoader();
				tensorInfos = safetensorsLoader.ReadTensorsInfoFromFile(ModelPath);
			}
			else
			{
				throw new ArgumentException("Invalid file extension");
			}

			if (tensorInfos.Count(a => a.Name.Contains("model.diffusion_model.double_blocks.")) > 0)
			{
				return ModelType.FLUX;
			}
			else if (tensorInfos.Count(a => a.Name.Contains("model.diffusion_model.joint_blocks.")) > 0)
			{
				return ModelType.SD3;
			}
			else if (tensorInfos.Count(a => a.Name.Contains("conditioner.embedders.1")) > 0)
			{
				return ModelType.SDXL;
			}
			else
			{
				return ModelType.SD1;
			}

		}


	}
}
