using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace StableDiffusionSharp.Scheduler
{
	internal class DiscreteSchedule : Module
	{
		private Tensor sigmas;
		private Tensor log_sigmas;
		private bool quantize;

		public DiscreteSchedule(Tensor sigmas, bool quantize = false) : base(nameof(DiscreteSchedule))
		{
			this.sigmas = sigmas;
			log_sigmas = sigmas.log();
			this.quantize = quantize;
			RegisterComponents();
		}

		public Tensor sigma_mix => sigmas.max();
		public Tensor sigma_max => sigmas.min();

		public Tensor t_to_sigma(Tensor t)
		{
			t = t.@float();
			Tensor low_idx = t.floor().@long();
			Tensor high_idx = t.ceil().@long();
			Tensor w = t.frac();
			Tensor log_sigma = (1 - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx];
			return log_sigma.exp();
		}

		public Tensor sigma_to_t(Tensor sigma, bool? quantize = null)
		{
			quantize = quantize ?? this.quantize;
			Tensor log_sigma = sigma.log();
			Tensor dists = log_sigma - log_sigmas[.., TensorIndex.None];

			if (quantize == true)
			{
				return dists.abs().argmin(dim: 0).view(sigma.shape);
			}

			Tensor low_idx = dists.ge(0).cumsum(dim: 0).argmax(dim: 0).clamp(max: log_sigmas.shape[0] - 2);
			Tensor high_idx = low_idx + 1;
			var (low, high) = (log_sigmas[low_idx], log_sigmas[high_idx]);
			Tensor w = (low - log_sigma) / (low - high);
			w = w.clamp(0, 1);
			Tensor t = (1 - w) * low_idx + w * high_idx;
			return t.view(sigma.shape);
		}
	}
}
