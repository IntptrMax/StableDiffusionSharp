using TorchSharp;
using static TorchSharp.torch;

namespace StableDiffusionSharp.Sampler
{
	public abstract class BasicSampler
	{
		public Tensor Sigmas;
		internal Tensor Timesteps;
		private Scheduler.DiscreteSchedule schedule;
		private readonly TimestepSpacing timestepSpacing;

		public BasicSampler(int num_train_timesteps = 1000, float beta_start = 0.00085f, float beta_end = 0.012f, int steps_offset = 1, TimestepSpacing timestepSpacing = TimestepSpacing.Leading)
		{
			this.timestepSpacing = timestepSpacing;
			Tensor betas = GetBetaSchedule(beta_start, beta_end, num_train_timesteps);
			Tensor alphas = 1.0f - betas;
			Tensor alphas_cumprod = torch.cumprod(alphas, 0);
			this.Sigmas = torch.pow((1.0f - alphas_cumprod) / alphas_cumprod, 0.5f);
		}

		public Tensor InitNoiseSigma()
		{
			if (timestepSpacing == TimestepSpacing.Linspace || timestepSpacing == TimestepSpacing.Trailing)
			{
				return Sigmas.max();
			}
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
			//long t_max = Sigmas.NumberOfElements - 1;
			//this.Timesteps = torch.linspace(t_max, 0, num_inference_steps);
			this.Timesteps = GetTimeSteps(Sigmas.NumberOfElements, num_inference_steps, timestepSpacing);
			schedule = new Scheduler.DiscreteSchedule(Sigmas);
			this.Sigmas = append_zero(schedule.t_to_sigma(this.Timesteps));
		}

		private Tensor GetTimeSteps(double t_max, long num_steps, TimestepSpacing timestepSpacing)
		{
			if (timestepSpacing == TimestepSpacing.Linspace)
			{
				return torch.linspace(t_max - 1, 0, num_steps);
			}
			else if (timestepSpacing == TimestepSpacing.Leading)
			{
				long step_ratio = (long)t_max / num_steps;
				return torch.linspace(t_max - step_ratio, 0, num_steps) + 1;
			}
			else
			{
				long step_ratio = (long)t_max / num_steps;
				return torch.arange(t_max, 0, -step_ratio).round() - 1;
			}
		}

		public abstract Tensor Step(torch.Tensor model_output, int step_index, torch.Tensor sample, long seed = 0, float s_churn = 0, float s_tmin = 0, float s_tmax = float.PositiveInfinity, float s_noise = 1);			
		
		private Tensor GetBetaSchedule(float beta_start, float beta_end, int num_train_timesteps)
		{
			return torch.pow(torch.linspace(Math.Pow(beta_start, 0.5), Math.Pow(beta_end, 0.5), num_train_timesteps, ScalarType.Float32), 2);
		}

		private static Tensor append_zero(Tensor x)
		{
			return torch.cat(new Tensor[] { x, x.new_zeros(1) });
		}
	}
}
