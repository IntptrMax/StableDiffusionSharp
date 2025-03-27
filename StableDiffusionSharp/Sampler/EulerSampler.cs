using TorchSharp;
using static TorchSharp.torch;

namespace StableDiffusionSharp.Sampler
{
	internal class EulerSampler : BasicSampler
	{
		public EulerSampler(int num_train_timesteps = 1000, float beta_start = 0.00085f, float beta_end = 0.012f, int steps_offset = 1) : base(num_train_timesteps, beta_start, beta_end, steps_offset)
		{

		}

		public override torch.Tensor Step(torch.Tensor model_output, int step_index, torch.Tensor sample, long seed = 0, float s_churn = 0, float s_tmin = 0, float s_tmax = float.PositiveInfinity, float s_noise = 1)
		{
			Generator generator = torch.manual_seed(seed);
			torch.set_rng_state(generator.get_state());
			float sigma = base.Sigmas[step_index].ToSingle();
			float gamma = s_tmin <= sigma && sigma <= s_tmax ? (float)Math.Min(s_churn / (Sigmas.NumberOfElements - 1f), Math.Sqrt(2.0f) - 1.0f) : 0f;
			Tensor noise = torch.randn_like(model_output);
			Tensor epsilon = noise * s_noise;
			float sigma_hat = sigma * (gamma + 1.0f);
			if (gamma > 0)
			{
				sample = sample + epsilon * (float)Math.Sqrt(Math.Pow(sigma_hat, 2f) - Math.Pow(sigma, 2f));
			}
			Tensor pred_original_sample = sample - sigma_hat * model_output;  // to_d and sigma is c_out
			Tensor derivative = (sample - pred_original_sample) / sigma_hat;
			Tensor dt = Sigmas[step_index + 1] - sigma_hat;
			return sample + derivative * dt;
		}
	}
}
