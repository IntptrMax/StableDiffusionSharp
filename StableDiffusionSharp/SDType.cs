namespace StableDiffusionSharp
{
	public enum SDScalarType
	{
		Float16 = 5,
		Float32 = 6,
		BFloat16 = 15,
	}

	public enum SDDeviceType
	{
		CPU = 0,
		CUDA = 1,
	}

	public enum SDSamplerType
	{
		EulerAncestral = 0,
		Euler = 1,
	}

	public enum ModelType
	{
		SD1,
		SD2,
		SD3,
		SDXL,
		FLUX,
	}

}
