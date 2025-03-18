using StableDiffusionSharp.ModelLoader;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace StableDiffusionSharp
{
	public class Esrgan : IDisposable
	{
		private readonly RRDBNet rrdbnet;
		private readonly Device device;
		private readonly ScalarType dtype;
		public Esrgan(int num_block = 23, SDDeviceType deviceType = SDDeviceType.CUDA, SDScalarType scalarType = SDScalarType.Float16)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			device = new Device((DeviceType)deviceType);
			dtype = (ScalarType)scalarType;
			rrdbnet = new RRDBNet(num_in_ch: 3, num_out_ch: 3, num_feat: 64, num_block: num_block, num_grow_ch: 32, scale: 4);
			rrdbnet.to(device, dtype);
		}

		/// <summary>
		/// Residual Dense Block.
		/// </summary>
		private class ResidualDenseBlock : Module<Tensor, Tensor>
		{
			private readonly Conv2d conv1;
			private readonly Conv2d conv2;
			private readonly Conv2d conv3;
			private readonly Conv2d conv4;
			private readonly Conv2d conv5;
			private readonly LeakyReLU lrelu;

			/// <summary>
			/// Used in RRDB block in ESRGAN.
			/// </summary>
			/// <param name="num_feat">Channel number of intermediate features.</param>
			/// <param name="num_grow_ch">Channels for each growth.</param>
			public ResidualDenseBlock(int num_feat = 64, int num_grow_ch = 32) : base(nameof(ResidualDenseBlock))
			{
				this.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1);
				this.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1);
				this.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1);
				this.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1);
				this.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1);
				this.lrelu = nn.LeakyReLU(negative_slope: 0.2f, inplace: true);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using (NewDisposeScope())
				{
					Tensor x1 = this.lrelu.forward(this.conv1.forward(x));
					Tensor x2 = this.lrelu.forward(this.conv2.forward(torch.cat([x, x1], 1)));
					Tensor x3 = this.lrelu.forward(this.conv3.forward(torch.cat([x, x1, x2], 1)));
					Tensor x4 = this.lrelu.forward(this.conv4.forward(torch.cat([x, x1, x2, x3], 1)));
					Tensor x5 = this.conv5.forward(torch.cat([x, x1, x2, x3, x4], 1));
					// Empirically, we use 0.2 to scale the residual for better performance
					return (x5 * 0.2 + x).MoveToOuterDisposeScope();
				}
			}
		}

		/// <summary>
		/// Residual in Residual Dense Block.
		/// </summary>
		private class RRDB : Module<Tensor, Tensor>
		{
			private readonly ResidualDenseBlock rdb1;
			private readonly ResidualDenseBlock rdb2;
			private readonly ResidualDenseBlock rdb3;

			/// <summary>
			/// Used in RRDB-Net in ESRGAN.
			/// </summary>
			/// <param name="num_feat">Channel number of intermediate features.</param>
			/// <param name="num_grow_ch">Channels for each growth.</param>
			public RRDB(int num_feat, int num_grow_ch = 32) : base(nameof(RRDB))
			{
				this.rdb1 = new ResidualDenseBlock(num_feat, num_grow_ch);
				this.rdb2 = new ResidualDenseBlock(num_feat, num_grow_ch);
				this.rdb3 = new ResidualDenseBlock(num_feat, num_grow_ch);
				RegisterComponents();
			}
			public override Tensor forward(Tensor x)
			{
				using (NewDisposeScope())
				{
					Tensor @out = this.rdb1.forward(x);
					@out = this.rdb2.forward(@out);
					@out = this.rdb3.forward(@out);
					// Empirically, we use 0.2 to scale the residual for better performance
					return (@out * 0.2 + x).MoveToOuterDisposeScope();
				}
			}
		}

		private class RRDBNet : Module<Tensor, Tensor>
		{
			private readonly int scale;
			private readonly Conv2d conv_first;
			private readonly Sequential body;
			private readonly Conv2d conv_body;
			private readonly Conv2d conv_up1;
			private readonly Conv2d conv_up2;
			private readonly Conv2d conv_hr;
			private readonly Conv2d conv_last;
			private readonly LeakyReLU lrelu;

			public RRDBNet(int num_in_ch, int num_out_ch, int scale = 4, int num_feat = 64, int num_block = 23, int num_grow_ch = 32) : base(nameof(RRDBNet))
			{
				this.scale = scale;
				if (scale == 2)
				{
					num_in_ch = num_in_ch * 4;
				}
				else if (scale == 1)
				{
					num_in_ch = num_in_ch * 16;
				}
				this.conv_first = Conv2d(num_in_ch, num_feat, 3, 1, 1);
				this.body = Sequential();
				for (int i = 0; i < num_block; i++)
				{
					body.append(new RRDB(num_feat: num_feat, num_grow_ch: num_grow_ch));
				}
				this.conv_body = Conv2d(num_feat, num_feat, 3, 1, 1);
				// upsample
				this.conv_up1 = Conv2d(num_feat, num_feat, 3, 1, 1);
				this.conv_up2 = Conv2d(num_feat, num_feat, 3, 1, 1);
				this.conv_hr = Conv2d(num_feat, num_feat, 3, 1, 1);
				this.conv_last = Conv2d(num_feat, num_out_ch, 3, 1, 1);
				this.lrelu = LeakyReLU(negative_slope: 0.2f, inplace: true);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using (NewDisposeScope())
				{
					Tensor feat = x;
					if (this.scale == 2)
					{
						feat = pixel_unshuffle(x, scale: 2);
					}
					else if (this.scale == 1)
					{
						feat = pixel_unshuffle(x, scale: 4);
					}
					feat = this.conv_first.forward(feat);
					Tensor body_feat = this.conv_body.forward(this.body.forward(feat));
					feat = feat + body_feat;
					// upsample
					feat = this.lrelu.forward(this.conv_up1.forward(torch.nn.functional.interpolate(feat, scale_factor: [2, 2], mode: InterpolationMode.Nearest)));
					feat = this.lrelu.forward(this.conv_up2.forward(torch.nn.functional.interpolate(feat, scale_factor: [2, 2], mode: InterpolationMode.Nearest)));
					Tensor @out = this.conv_last.forward(this.lrelu.forward(this.conv_hr.forward(feat)));
					return @out.MoveToOuterDisposeScope();
				}
			}

			/// <summary>
			/// Pixel unshuffle.
			/// </summary>
			/// <param name="x">Input feature with shape (b, c, hh, hw).</param>
			/// <param name="scale">Downsample ratio.</param>
			/// <returns>the pixel unshuffled feature.</returns>
			private Tensor pixel_unshuffle(Tensor x, int scale)
			{
				long b = x.shape[0];
				long c = x.shape[1];
				long hh = x.shape[2];
				long hw = x.shape[3];

				long out_channel = c * (scale * scale);

				if (hh % scale != 0 && hw % scale != 0)
				{
					throw new ArgumentException("Width or Hight are not match");
				}

				long h = hh / scale;
				long w = hw / scale;

				Tensor x_view = x.view(b, c, h, scale, w, scale);
				return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w);
			}
		}

		public void LoadModel(string path)
		{
			Dictionary<string, Tensor> state_dict = PickleLoader.Load(path);
			var (miss, unexp) = rrdbnet.load_state_dict(state_dict);
			rrdbnet.eval();
		}

		public ImageMagick.MagickImage UpScale(ImageMagick.MagickImage inputImg)
		{
			using (torch.no_grad())
			{
				Tensor tensor = Tools.GetTensorFromImage(inputImg);
				tensor = tensor.unsqueeze(0) / 255.0;
				tensor = tensor.to(dtype, device);
				Tensor op = rrdbnet.forward(tensor);
				op = (op.cpu() * 255.0f).clamp(0, 255).@byte();
				return Tools.GetImageFromTensor(op);
			}
		}

		public void Dispose()
		{
			rrdbnet?.Dispose();
		}
	}
}
