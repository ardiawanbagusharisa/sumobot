using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace ML.LanguageModels
{
    public class BPETokenizer
    {
        public Dictionary<string, int> vocab;
        public Dictionary<int, string> idToToken;
        private List<(string, string)> merges;
        private Dictionary<(string, string), int> ranks;
        private Dictionary<byte, char> byteEncoder;
        private Dictionary<char, byte> byteDecoder;
        private bool addPrefixSpace;

        public BPETokenizer(string tokenizerJsonPath)
        {
            string json = System.IO.File.ReadAllText(tokenizerJsonPath);
            JObject root = JObject.Parse(json);

            // --- Load vocab ---
            var vocabObj = (JObject)root["model"]["vocab"];
            vocab = vocabObj.Properties().ToDictionary(p => p.Name, p => (int)p.Value);
            idToToken = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);

            // --- Load merges ---
            var mergesArray = (JArray)root["model"]["merges"];
            merges = mergesArray.Select(line =>
            {
                // var parts = line.ToString().Split(' ');
                return ((string)line[0], (string)line[1]);
            }).ToList();
            ranks = merges.Select((pair, idx) => new { pair, idx })
                          .ToDictionary(x => x.pair, x => x.idx);

            // --- Byte encoder / decoder ---
            (byteEncoder, byteDecoder) = GetByteEncoderDecoder();

            // --- Settings ---
            addPrefixSpace = root["pre_tokenizer"]?["add_prefix_space"]?.ToObject<bool>() ?? false;
        }

        private (Dictionary<byte, char>, Dictionary<char, byte>) GetByteEncoderDecoder()
        {
            // Matches HuggingFace GPT-2 / ByteLevel encoding
            var bs = Enumerable.Range('!', '~' - '!' + 1)
                .Concat(Enumerable.Range(161, 255 - 161 + 1))
                .ToList();

            var cs = bs.Select(b => (char)b).ToList();
            int n = 0;
            for (int b = 0; b < 256; b++)
            {
                if (!bs.Contains(b))
                {
                    bs.Add(b);
                    cs.Add((char)(256 + n));
                    n++;
                }
            }

            var byteEncoder = new Dictionary<byte, char>();
            var byteDecoder = new Dictionary<char, byte>();

            for (int i = 0; i < 256; i++)
            {
                byteEncoder[(byte)bs[i]] = cs[i];
                byteDecoder[cs[i]] = (byte)bs[i];
            }

            return (byteEncoder, byteDecoder);
        }

        private string ByteEncodeText(string text)
        {
            var bytes = Encoding.UTF8.GetBytes(text);
            return new string(bytes.Select(b => byteEncoder[b]).ToArray());
        }

        private string ByteDecodeText(string mapped)
        {
            var bytes = mapped.Select(c => byteDecoder[c]).ToArray();
            return Encoding.UTF8.GetString(bytes);
        }

        private List<string> ApplyBPE(string token)
        {
            if (token.Length == 1)
                return new List<string> { token };

            var word = token.Select(c => c.ToString()).ToList();
            var pairs = GetPairs(word);

            while (pairs.Count > 0)
            {
                var bigram = pairs
                    .OrderBy(p => ranks.ContainsKey(p) ? ranks[p] : int.MaxValue)
                    .First();

                if (!ranks.ContainsKey(bigram))
                    break;

                string first = bigram.Item1;
                string second = bigram.Item2;
                var newWord = new List<string>();

                int i = 0;
                while (i < word.Count)
                {
                    int j = word.IndexOf(first, i);
                    if (j == -1)
                    {
                        newWord.AddRange(word.Skip(i));
                        break;
                    }
                    newWord.AddRange(word.Skip(i).Take(j - i));
                    i = j;

                    if (i < word.Count - 1 && word[i] == first && word[i + 1] == second)
                    {
                        newWord.Add(first + second);
                        i += 2;
                    }
                    else
                    {
                        newWord.Add(word[i]);
                        i++;
                    }
                }

                word = newWord;
                if (word.Count == 1) break;
                pairs = GetPairs(word);
            }

            return word; // <--- now returns multiple tokens
        }


        private HashSet<(string, string)> GetPairs(List<string> word)
        {
            var pairs = new HashSet<(string, string)>();
            string prevChar = word[0];
            for (int i = 1; i < word.Count; i++)
            {
                pairs.Add((prevChar, word[i]));
                prevChar = word[i];
            }
            return pairs;
        }

        public int[] Encode(string text)
        {
            if (addPrefixSpace && !string.IsNullOrEmpty(text) && text[0] != ' ')
                text = " " + text;

            // ByteLevel mapping
            string mapped = ByteEncodeText(text);

            // Regex pre-tokenization
            var matches = Regex.Matches(mapped, @"\S+\s*");

            var tokens = new List<int>();
            foreach (Match match in matches)
            {
                string part = match.Value;
                var bpeTokens = ApplyBPE(part); // now returns a list

                foreach (var subToken in bpeTokens)
                {
                    if (!vocab.ContainsKey(subToken))
                        throw new Exception($"Token '{subToken}' not in vocab.");
                    tokens.Add(vocab[subToken]);
                }
            }
            return tokens.ToArray();
        }


        public string Decode(List<int> ids)
        {
            var mapped = string.Concat(ids.Select(id => idToToken[id]));
            return ByteDecodeText(mapped);
        }
    }
}