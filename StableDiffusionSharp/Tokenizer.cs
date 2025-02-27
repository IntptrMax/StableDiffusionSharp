﻿using Microsoft.ML.Tokenizers;
using StableDiffusionSharp.Properties;
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
				string path = Path.GetDirectoryName(vocabPath);
				if (!Directory.Exists(path))
				{
					Directory.CreateDirectory(path);
				}
				byte[] vocabBytes = (byte[])Resources.ResourceManager.GetObject("vocab.json");
				File.WriteAllBytes(vocabPath, vocabBytes);
			}

			if (!File.Exists(mergesPath))
			{
				string path = Path.GetDirectoryName(mergesPath);
				if (!Directory.Exists(path))
				{
					Directory.CreateDirectory(path);
				}
				byte[] mergesBytes = (byte[])Resources.ResourceManager.GetObject("merges.txt");
				File.WriteAllBytes(mergesPath, mergesBytes);
			}


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
