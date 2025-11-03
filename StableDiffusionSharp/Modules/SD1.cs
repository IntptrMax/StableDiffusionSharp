using static StableDiffusionSharp.Modules.UNet;
using static TorchSharp.torch;

namespace StableDiffusionSharp.Modules
{
	internal class SD1 : SDModel
	{
		public SD1(float scale_factor = 0.18215f, int in_channels = 4, int model_channels = 320, int context_dim = 768, int num_head = 8, float dropout = 0.0f, int embed_dim = 4, bool double_z = true, int z_channels = 4, Device? device = null, ScalarType? dtype = null) : base(scale_factor: scale_factor, device: device, dtype: dtype)
		{
			this.cliper = new Clip.SD1_5Cliper(device: device, dtype: dtype);
			this.diffusion = new SDUnet(model_channels, in_channels, num_head, context_dim, dropout, device: device, dtype: dtype);
			this.decoder = new VAE.Decoder(embed_dim: embed_dim, z_channels: z_channels, device: device, dtype: dtype);
			this.encoder = new VAE.Encoder(embed_dim: embed_dim, z_channels: z_channels, double_z: double_z, device: device, dtype: dtype);
			this.vaeApprox = new VAEApprox(4, device, dtype);
			this.vaeApproxPath = @".\Models\VAEApprox\vaeapp_sd15.pth";
		}
	}

}

