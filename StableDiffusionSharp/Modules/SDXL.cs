using static StableDiffusionSharp.Modules.UNet;
using static TorchSharp.torch;

namespace StableDiffusionSharp.Modules
{
	internal class SDXL : SDModel
	{
		public SDXL(float scale_factor = 0.13025f, int in_channels = 4, int model_channels = 320, int context_dim = 2048, int num_head = 20, float dropout = 0.0f, int adm_in_channels = 2816, int embed_dim = 4, int z_channels = 4, Device? device = null, bool double_z = true, ScalarType? dtype = null) : base(scale_factor: scale_factor, device: device, dtype: dtype)
		{
			this.cliper = new Clip.SDXLCliper(device: CPU, dtype: ScalarType.Float32);
			this.diffusion = new SDXLUnet(model_channels, in_channels, num_head, context_dim, adm_in_channels, dropout, device: device, dtype: dtype);
			this.decoder = new VAE.Decoder(embed_dim: embed_dim, z_channels: z_channels, device: device, dtype: dtype);
			this.encoder = new VAE.Encoder(embed_dim: embed_dim, z_channels: z_channels, double_z: double_z, device: CPU, dtype: ScalarType.Float32);
			this.vaeApprox = new VAEApprox(4, device, dtype);
			this.vaeApproxPath = @".\Models\VAEApprox\xlvaeapp.pth";
		}
	}
}

