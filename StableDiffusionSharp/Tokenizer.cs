using Microsoft.ML.Tokenizers;
using TorchSharp;
using static TorchSharp.torch;

namespace StableDiffusionSharp
{
	public class Tokenizer
	{
		private readonly BpeTokenizer _tokenizer;
		private readonly int _startToken;
		private readonly int _endToken;

		public Tokenizer(string vocabPath, string mergesPath, int startToken = 49406, int endToken = 49407)
		{
			_tokenizer = BpeTokenizer.Create(vocabPath, mergesPath, endOfWordSuffix: "</w>");
			_startToken = startToken;
			_endToken = endToken;
		}

		public Tensor Tokenize(string text, int maxTokens = 77)
		{
			var res = _tokenizer.EncodeToIds(text);
			var tokens = new[] { _startToken }.Concat(res.Concat(Enumerable.Repeat(0, maxTokens - res.Count - 2))).Concat(new[] { _endToken }).ToArray();
			return torch.tensor(tokens, @long).unsqueeze(0);
		}
	}
}
