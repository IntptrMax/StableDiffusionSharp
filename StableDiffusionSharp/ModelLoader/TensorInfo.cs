namespace StableDiffusionSharp.ModelLoader
{
	internal class TensorInfo
	{
		public string Name { get; set; }
		public TorchSharp.torch.ScalarType Type { get; set; } = TorchSharp.torch.ScalarType.Float16;
		public List<long> Shape { get; set; } = new List<long>();
		public List<ulong> Stride { get; set; } = new List<ulong>();
		public string DataNameInZipFile { get; set; }
		public string FileName { get; set; }
		public List<ulong> Offset { get; set; } = new List<ulong>();
		public long BodyPosition { get; set; }
	}
}
