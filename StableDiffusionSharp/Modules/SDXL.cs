using TorchSharp;
using static TorchSharp.torch;

namespace StableDiffusionSharp.Modules
{
	public class SDXL : SD1
	{
		public SDXL(Device? device = null, ScalarType? dtype = null) : base(device, dtype)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			this.device = device ?? torch.CPU;
			this.dtype = dtype ?? torch.float32;

			this.scale_factor = 0.13025f;

			this.in_channels = 4;
			this.model_channels = 320;
			this.context_dim = 2048;
			this.num_head = 20;
			this.dropout = 0.0f;
			this.adm_in_channels = 2816;

			this.embed_dim = 4;
			this.double_z = true;
			this.z_channels = 4;
		}
	}
}

