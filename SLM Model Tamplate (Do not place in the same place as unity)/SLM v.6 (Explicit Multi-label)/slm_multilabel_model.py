import torch
import torch.nn as nn

class SLM_Multilabel_BiLSTM(nn.Module):
    def __init__(self, vocab_in_size, num_actions, emb_dim=128, hidden_dim=256, num_layers=1, pad_idx_in=0):
        super().__init__()
        self.embedding_in = nn.Embedding(vocab_in_size, emb_dim, padding_idx=pad_idx_in)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc_out = nn.Linear(hidden_dim * 2, num_actions)  # BiLSTM, so hidden*2

    def forward(self, src):
        # src: (B, seq_len)
        emb = self.embedding_in(src)                  # (B, seq_len, emb_dim)
        _, (h, c) = self.encoder(emb)                 # h: (num_layers*2, B, hidden_dim)
        # Concatenate last fwd & bwd
        h_cat = torch.cat([h[-2], h[-1]], dim=1)      # (B, hidden_dim*2)
        logits = self.fc_out(h_cat)                   # (B, num_actions)
        return logits                                 # Logits, will use sigmoid later

class SLM_Multilabel_Transformer(nn.Module):
    def __init__(self, vocab_in_size, num_actions, emb_dim=128, nhead=4, num_layers=2, pad_idx_in=0):
        super().__init__()
        self.embedding_in = nn.Embedding(vocab_in_size, emb_dim, padding_idx=pad_idx_in)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=emb_dim*2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(emb_dim, num_actions)

    def forward(self, src):
        emb = self.embedding_in(src)                 # (B, seq_len, emb_dim)
        enc_out = self.encoder(emb)                  # (B, seq_len, emb_dim)
        # Mean pooling over seq_len
        enc_vec = enc_out.mean(dim=1)                # (B, emb_dim)
        logits = self.fc_out(enc_vec)                # (B, num_actions)
        return logits

# ========== Utility: Test Instantiation ==========
if __name__ == "__main__":
    batch = 3
    seq_len = 24
    vocab_in_size = 100
    num_actions = 8
    pad_idx_in = 0

    print("Testing Multi-label BiLSTM:")
    model = SLM_Multilabel_BiLSTM(vocab_in_size, num_actions, pad_idx_in=pad_idx_in)
    x = torch.randint(0, vocab_in_size, (batch, seq_len))
    out = model(x)
    print("Output shape:", out.shape)   # (batch, num_actions)

    print("Testing Multi-label Transformer:")
    model2 = SLM_Multilabel_Transformer(vocab_in_size, num_actions, pad_idx_in=pad_idx_in)
    out2 = model2(x)
    print("Output shape:", out2.shape)  # (batch, num_actions)
