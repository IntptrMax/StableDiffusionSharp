using TorchSharp;
using static TorchSharp.torch;

namespace StableDiffusionSharp.Sampler
{
	internal class EulerAncestralSampler : BasicSampler
	{
		public EulerAncestralSampler(int num_train_timesteps = 1000, float beta_start = 0.00085f, float beta_end = 0.012f, int steps_offset = 1) : base(num_train_timesteps, beta_start, beta_end, steps_offset)
		{

		}
		public override torch.Tensor Step(torch.Tensor model_output, int step_index, torch.Tensor sample, long seed = 0, float s_churn = 0, float s_tmin = 0, float s_tmax = float.PositiveInfinity, float s_noise = 1)
		{
			Generator generator = torch.manual_seed(seed);
			torch.set_rng_state(generator.get_state());


			float sigma = base.Sigmas[step_index].ToSingle();

			Tensor predOriginalSample = sample - model_output * sigma;
			Tensor sigmaFrom = base.Sigmas[step_index];
			Tensor sigmaTo = base.Sigmas[step_index + 1];
			Tensor sigmaFromLessSigmaTo = torch.pow(sigmaFrom, 2) - torch.pow(sigmaTo, 2);
			Tensor sigmaUpResult = torch.pow(sigmaTo, 2) * sigmaFromLessSigmaTo / torch.pow(sigmaFrom, 2);

			Tensor sigmaUp = sigmaUpResult.ToSingle() < 0 ? -torch.pow(torch.abs(sigmaUpResult), 0.5f) : torch.pow(sigmaUpResult, 0.5f);
			Tensor sigmaDownResult = torch.pow(sigmaTo, 2) - torch.pow(sigmaUp, 2);
			Tensor sigmaDown = sigmaDownResult.ToSingle() < 0 ? -torch.pow(torch.abs(sigmaDownResult), 0.5f) : torch.pow(sigmaDownResult, 0.5f);
			Tensor derivative = (sample - predOriginalSample) / sigma;   // to_d and sigma is c_out
			Tensor delta = sigmaDown - sigma;
			Tensor prevSample = sample + derivative * delta;
			var noise = torch.randn_like(prevSample);
			prevSample = prevSample + noise * sigmaUp;
			return prevSample;
		}
	}
}
