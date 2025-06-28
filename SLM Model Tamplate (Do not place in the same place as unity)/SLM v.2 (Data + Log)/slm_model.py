import torch
import torch.nn as nn

class SimpleTokenizer:
    """A simple, word-level tokenizer."""
    def __init__(self, vocab=None):
        self.word2idx = {}
        self.idx2word = {}
        self.unk = "<unk>"
        if vocab:
            self.build_vocab(vocab)
    
    def build_vocab(self, texts):
        words = set()
        for text in texts:
            words.update(text.lower().split())
        words = sorted(list(words))
        self.word2idx = {w: i+2 for i, w in enumerate(words)}
        self.word2idx["<pad>"] = 0
        self.word2idx[self.unk] = 1
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def encode(self, text, maxlen):
        tokens = text.lower().split()
        ids = [self.word2idx.get(tok, self.word2idx[self.unk]) for tok in tokens]
        # Padding
        if len(ids) < maxlen:
            ids += [0] * (maxlen - len(ids))
        return ids[:maxlen]

    def decode(self, ids):
        return [self.idx2word.get(i, self.unk) for i in ids]

class SLMModel(nn.Module):
    """LSTM encoder for SLM, output to classifier"""
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
       # For multi-label: use sigmoid, for multi-class: use softmax in loss

    def forward(self, x):
        emb = self.embedding(x)                   # (B, T, emb_dim)
        _, (h_n, _) = self.encoder(emb)           # h_n: (1, B, hidden_dim)
        h = h_n.squeeze(0)                        # (B, hidden_dim)
        out = self.fc(h)                          # (B, output_dim)
        return out
