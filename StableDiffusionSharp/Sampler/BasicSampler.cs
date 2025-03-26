using TorchSharp;
using static TorchSharp.torch;

namespace StableDiffusionSharp.Sampler
{
	public abstract class BasicSampler
	{
		public Tensor Sigmas;
		internal Tensor Timesteps;
		private Scheduler.DiscreteSchedule schedule;

		public BasicSampler(int num_train_timesteps = 1000, float beta_start = 0.00085f, float beta_end = 0.012f, int steps_offset = 1)
		{
			Tensor betas = GetBetaSchedule(beta_start, beta_end, num_train_timesteps);
			Tensor alphas = 1.0f - betas;
			Tensor alphas_cumprod = torch.cumprod(alphas, 0);
			this.Sigmas = torch.pow((1.0f - alphas_cumprod) / alphas_cumprod, 0.5f);
		}

		public Tensor InitNoiseSigma()
		{
			return torch.sqrt(torch.pow(Sigmas.max(), 2) + 1);
		}

		public Tensor ScaleModelInput(Tensor sample, int step_index)
		{
			Tensor sigma = Sigmas[step_index];
			return sample / torch.sqrt(torch.pow(sigma, 2) + 1);
		}

		/// <summary>
		/// Get the scalings for the given step index
		/// </summary>
		/// <param name="step_index"></param>
		/// <returns>Tensor c_out, Tensor c_in</returns>
		public (Tensor, Tensor) GetScalings(int step_index)
		{
			Tensor sigma = Sigmas[step_index];
			Tensor c_out = -sigma;
			Tensor c_in = 1 / torch.sqrt(torch.pow(sigma, 2) + 1);
			return (c_out, c_in);
		}
		public Tensor append_dims(Tensor x, long target_dims)
		{
			long dims_to_append = target_dims - x.ndim;
			if (dims_to_append < 0)
			{
				throw new ArgumentException("target_dims must be greater than x.ndim");
			}
			long[] dims = x.shape;
			for (int i = 0; i < dims_to_append; i++)
			{
				dims.Append(1);
			}
			return x.view(dims);
		}



		public void SetTimesteps(long num_inference_steps)
		{
			if (num_inference_steps < 1)
			{
				throw new ArgumentException("num_inference_steps must be greater than 0");
			}
			long t_max = Sigmas.NumberOfElements - 1;
			this.Timesteps = torch.linspace(t_max, 0, num_inference_steps);
			schedule = new Scheduler.DiscreteSchedule(Sigmas);
			this.Sigmas = append_zero(schedule.t_to_sigma(this.Timesteps));
		}

		public virtual Tensor Step(Tensor model_output, int step_index, Tensor sample, long seed = 0, float s_churn = 0.0f, float s_tmin = 0.0f, float s_tmax = float.PositiveInfinity, float s_noise = 1.0f)
		{
			// It is the same as EulerSampler
			Generator generator = torch.manual_seed(seed);
			torch.set_rng_state(generator.get_state());
			float sigma = Sigmas[step_index].ToSingle();
			float gamma = s_tmin <= sigma && sigma <= s_tmax ? (float)Math.Min(s_churn / (Sigmas.NumberOfElements - 1f), Math.Sqrt(2.0f) - 1.0f) : 0f;
			Tensor epsilon = torch.randn_like(model_output) * s_noise;
			float sigma_hat = sigma * (gamma + 1);
			if (gamma > 0)
			{
				sample = sample + epsilon * (float)Math.Sqrt(Math.Pow(sigma_hat, 2f) - Math.Pow(sigma, 2f));
			}
			Tensor pred_original_sample = sample - sigma_hat * model_output;
			Tensor derivative = (sample - pred_original_sample) / sigma_hat;
			float dt = Sigmas[step_index + 1].ToSingle() - sigma_hat;
			return sample + derivative * dt;
		}

		private Tensor GetBetaSchedule(float beta_start, float beta_end, int num_train_timesteps)
		{
			return torch.pow(torch.linspace(Math.Pow(beta_start, 0.5), Math.Pow(beta_end, 0.5), num_train_timesteps, ScalarType.Float32), 2);
		}

		private static Tensor append_zero(Tensor x)
		{
			return torch.cat([x, x.new_zeros([1])]);
		}
	}
}
