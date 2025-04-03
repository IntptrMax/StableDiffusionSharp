using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace StableDiffusionSharp.Modules
{
	internal class VAEApprox : Module<Tensor, Tensor>
	{
		private readonly Conv2d conv1;
		private readonly Conv2d conv2;
		private readonly Conv2d conv3;
		private readonly Conv2d conv4;
		private readonly Conv2d conv5;
		private readonly Conv2d conv6;
		private readonly Conv2d conv7;
		private readonly Conv2d conv8;

		internal VAEApprox(int latent_channels = 4, Device? device = null, ScalarType? dtype = null) : base(nameof(VAEApprox))
		{
			conv1 = Conv2d(latent_channels, 8, (7, 7), device: device, dtype: dtype);
			conv2 = Conv2d(8, 16, (5, 5), device: device, dtype: dtype);
			conv3 = Conv2d(16, 32, (3, 3), device: device, dtype: dtype);
			conv4 = Conv2d(32, 64, (3, 3), device: device, dtype: dtype);
			conv5 = Conv2d(64, 32, (3, 3), device: device, dtype: dtype);
			conv6 = Conv2d(32, 16, (3, 3), device: device, dtype: dtype);
			conv7 = Conv2d(16, 8, (3, 3), device: device, dtype: dtype);
			conv8 = Conv2d(8, 3, (3, 3), device: device, dtype: dtype);
			RegisterComponents();
		}

		public override Tensor forward(Tensor x)
		{
			using (NewDisposeScope())
			{
				int extra = 11;
				x = functional.interpolate(x, [x.shape[2] * 2, x.shape[3] * 2]);
				x = functional.pad(x, (extra, extra, extra, extra));

				foreach (var layer in ModuleList([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8]))
				{
					x = layer.forward(x);
					x = functional.leaky_relu(x, 0.1);
				}
				return x.MoveToOuterDisposeScope();
			}
		}

		public enum SharedModel
		{
			SD3,
			SDXL,
			SD
		}
	}
}
