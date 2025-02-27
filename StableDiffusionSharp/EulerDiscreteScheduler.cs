using TorchSharp;
using static TorchSharp.torch;

namespace StableDiffusionSharp
{
	internal class EulerDiscreteScheduler
	{
		private long num_train_timesteps_;
		private int steps_offset_;
		private Tensor betas_;
		private Tensor alphas_;
		private Tensor alphas_cumprod_;
		private Tensor sigmas_;
		public Tensor timesteps_;
		private long num_inference_steps_;

		internal EulerDiscreteScheduler(long num_train_timesteps = 1000, float beta_start = 0.00085f, float beta_end = 0.012f, int steps_offset = 1)
		{
			num_train_timesteps_ = num_train_timesteps;
			steps_offset_ = steps_offset;
			betas_ = torch.pow(torch.linspace(Math.Pow(beta_start, 0.5), Math.Pow(beta_end, 0.5), num_train_timesteps, ScalarType.Float32), 2);
			alphas_ = 1f - betas_;
			alphas_cumprod_ = torch.cumprod(alphas_, 0);
			var sigmas = torch.pow((1.0f - alphas_cumprod_) / alphas_cumprod_, 0.5f);
			sigmas_ = torch.cat(new Tensor[] { sigmas.flip(0), torch.tensor(new float[] { 0.0f }) });
			timesteps_ = torch.linspace(0, num_train_timesteps - 1, num_train_timesteps_).flip(0);
		}

		internal Tensor InitNoiseSigma()
		{
			return torch.pow(torch.pow(sigmas_.max(), 2) + 1, 0.5f);
		}

		internal Tensor ScaleModelInput(Tensor sample, Tensor timestep)
		{
			var step_index = (timesteps_ == timestep).nonzero().ToInt64();
			var sigma = sigmas_[step_index].ToSingle();
			sample = sample / (float)(Math.Pow(Math.Pow(sigma, 2) + 1, 0.5f));
			return sample;
		}

		internal void SetTimesteps(long num_inference_steps, torch.Device device)
		{
			num_inference_steps_ = num_inference_steps;
			long step_ratio = num_train_timesteps_ / num_inference_steps_;
			var timesteps = (torch.arange(0, num_inference_steps, ScalarType.Float32, device: device) * step_ratio).round().flip(0);
			timesteps = timesteps + steps_offset_;
			var sigmas = torch.pow((1.0f - alphas_cumprod_) / alphas_cumprod_, 0.5f);
			sigmas = Interp(timesteps, torch.arange(0, sigmas.shape[0], device: device), sigmas.to(device));
			sigmas_ = torch.cat(new Tensor[] { sigmas, torch.tensor(new float[] { 0.0f }, device: device) });
			timesteps_ = timesteps;
		}

		internal Tensor Step(Tensor model_output, Tensor timestep, Tensor sample)
		{
			var step_index = (timesteps_ == timestep).nonzero().ToInt64();
			var sigma = sigmas_[step_index].ToSingle();
			float gamma = 0;
			if (sigma >= 0 && sigma <= System.Single.PositiveInfinity)
			{
				gamma = (float)Math.Min(0f, Math.Pow(2, 0.5f) - 1);
			}
			var noise = torch.randn(model_output.shape, model_output.dtype, model_output.device);
			var sigma_hat = sigma * (gamma + 1);
			if (gamma > 0)
			{
				sample = sample + noise * (torch.pow(torch.pow(sigma_hat, 2) - torch.pow(sigma, 2), 0.5));
			}
			var pred_original_sample = sample - sigma_hat * model_output;
			var derivative = (sample - pred_original_sample) / sigma_hat;
			var dt = sigmas_[step_index + 1].ToSingle() - sigma_hat;
			var prev_sample = sample + derivative * dt;
			return prev_sample;
		}

		private Tensor Interp(Tensor x, Tensor xp, Tensor fp)
		{
			var sort_idx = torch.argsort(xp);
			xp = xp[sort_idx];
			fp = fp[sort_idx];
			var idx = torch.searchsorted(xp, x);
			idx = torch.clamp(idx, 0, xp.shape[0] - 2);
			var weight = (x - xp[idx]) / (xp[idx + 1] - xp[idx]);
			return fp[idx] * (1 - weight) + fp[idx + 1] * weight;
		}
	}
}
