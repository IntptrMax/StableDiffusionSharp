using Microsoft.ML.Tokenizers;
using System.Reflection;
using TorchSharp;
using static TorchSharp.torch;

namespace StableDiffusionSharp
{
	internal class Tokenizer
	{
		private readonly BpeTokenizer _tokenizer;
		private readonly int _startToken;
		private readonly int _endToken;

		public Tokenizer(string vocabPath, string mergesPath, int startToken = 49406, int endToken = 49407)
		{
			if (!File.Exists(vocabPath))
			{
				string path = Path.GetDirectoryName(vocabPath)!;
				if (!Directory.Exists(path))
				{
					Directory.CreateDirectory(path);
				}
				Assembly _assembly = Assembly.GetExecutingAssembly();
				string resourceName = "StableDiffusionSharp.Models.Clip.vocab.json";
				using (Stream stream = _assembly.GetManifestResourceStream(resourceName)!)
				{
					if (stream == null)
					{
						Console.WriteLine("Resource can't find!");
						return;
					}
					using (FileStream fileStream = new FileStream(vocabPath, FileMode.Create, FileAccess.Write))
					{
						stream.CopyTo(fileStream);
					}
				}

			}

			if (!File.Exists(mergesPath))
			{
				string path = Path.GetDirectoryName(mergesPath)!;
				if (!Directory.Exists(path))
				{
					Directory.CreateDirectory(path);
				}
				Assembly _assembly = Assembly.GetExecutingAssembly();
				string resourceName = "StableDiffusionSharp.Models.Clip.merges.txt";
				using (Stream stream = _assembly.GetManifestResourceStream(resourceName)!)
				{
					if (stream == null)
					{
						Console.WriteLine("Resource can't find!");
						return;
					}
					using (FileStream fileStream = new FileStream(mergesPath, FileMode.Create, FileAccess.Write))
					{
						stream.CopyTo(fileStream);
					}
				}

			}

			_tokenizer = BpeTokenizer.Create(vocabPath, mergesPath, endOfWordSuffix: "</w>");
			_startToken = startToken;
			_endToken = endToken;
		}

		public Tensor Tokenize(string text, int maxTokens = 77)
		{
			var res = _tokenizer.EncodeToIds(text).ToList();
			res.Insert(0, _startToken);
			res.Add(_endToken);
			return torch.tensor(res, ScalarType.Int64).unsqueeze(0);
		}
	}
}
