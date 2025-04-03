using TorchSharp;
using static TorchSharp.torch;

namespace StableDiffusionSharp.Modules
{
	public class SD1 : SDModel
	{
		public SD1(Device? device = null, ScalarType? dtype = null) : base(device, dtype)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			this.device = device ?? torch.CPU;
			this.dtype = dtype ?? torch.float32;

			// Default parameters
			this.scale_factor = 0.18215f;

			// UNet config
			this.in_channels = 4;
			this.model_channels = 320;
			this.context_dim = 768;
			this.num_head = 8;
			this.dropout = 0.0f;
			this.embed_dim = 4;

			// first stage config:
			this.embed_dim = 4;
			this.double_z = true;
			this.z_channels = 4;

		}
	}

}

