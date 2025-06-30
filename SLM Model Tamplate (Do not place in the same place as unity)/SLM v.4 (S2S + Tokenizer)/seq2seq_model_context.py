import torch
import torch.nn as nn

class Seq2SeqSLM_BiLSTM(nn.Module):
    def __init__(self, vocab_in_size, vocab_out_size, emb_dim=128, hidden_dim=256, num_layers=1, pad_idx_in=0, pad_idx_out=0):
        super().__init__()
        # ----- Encoder: BiLSTM -----
        self.embedding_in = nn.Embedding(vocab_in_size, emb_dim, padding_idx=pad_idx_in)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # ----- Decoder: Standard LSTM -----
        self.embedding_out = nn.Embedding(vocab_out_size, emb_dim, padding_idx=pad_idx_out)
        self.decoder = nn.LSTM(emb_dim, hidden_dim*2, num_layers, batch_first=True)  # hidden*2 (because BiLSTM)
        self.fc_out = nn.Linear(hidden_dim*2, vocab_out_size)

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5, max_len=16, start_token_idx=1):
        # src: (B, src_len), trg: (B, trg_len)
        batch_size = src.size(0)
        # ----- Encoder -----
        embedded_src = self.embedding_in(src)
        _, (h, c) = self.encoder(embedded_src)  # h, c: (num_layers*2, B, hidden)
        # Merge directions for decoder: concat last fwd & last bwd (flatten to shape compatible)
        h_cat = torch.cat([h[-2], h[-1]], dim=1).unsqueeze(0)  # (1, B, hidden*2)
        c_cat = torch.cat([c[-2], c[-1]], dim=1).unsqueeze(0)
        outputs = []
        if trg is not None:
            trg_len = trg.size(1)
            input_token = trg[:, 0].unsqueeze(1)
            hidden, cell = h_cat, c_cat
            for t in range(1, trg_len):
                emb = self.embedding_out(input_token)
                output, (hidden, cell) = self.decoder(emb, (hidden, cell))
                prediction = self.fc_out(output.squeeze(1))
                outputs.append(prediction.unsqueeze(1))
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                input_token = trg[:, t].unsqueeze(1) if (teacher_force and t < trg_len) else prediction.argmax(1, keepdim=True)
            outputs = torch.cat(outputs, dim=1)
            return outputs
        else:
            # Inference/generation
            input_token = torch.full((batch_size, 1), start_token_idx, dtype=torch.long, device=src.device)
            hidden, cell = h_cat, c_cat
            for _ in range(max_len):
                emb = self.embedding_out(input_token)
                output, (hidden, cell) = self.decoder(emb, (hidden, cell))
                prediction = self.fc_out(output.squeeze(1))
                outputs.append(prediction.unsqueeze(1))
                input_token = prediction.argmax(1, keepdim=True)
            outputs = torch.cat(outputs, dim=1)
            return outputs

# === BONUS: Mini Transformer Encoder ===
class Seq2SeqSLM_Transformer(nn.Module):
    def __init__(self, vocab_in_size, vocab_out_size, emb_dim=128, nhead=4, num_layers=2, pad_idx_in=0, pad_idx_out=0):
        super().__init__()
        # ----- Encoder -----
        self.embedding_in = nn.Embedding(vocab_in_size, emb_dim, padding_idx=pad_idx_in)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=emb_dim*2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # ----- Decoder -----
        self.embedding_out = nn.Embedding(vocab_out_size, emb_dim, padding_idx=pad_idx_out)
        self.decoder_lstm = nn.LSTM(emb_dim, emb_dim, num_layers=1, batch_first=True)
        self.fc_out = nn.Linear(emb_dim, vocab_out_size)

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5, max_len=16, start_token_idx=1):
        # src: (B, src_len), trg: (B, trg_len)
        batch_size = src.size(0)
        # ----- Encoder -----
        embedded_src = self.embedding_in(src)
        src_mask = None  # Optional: masking for pad tokens
        enc_out = self.encoder(embedded_src, mask=src_mask)  # (B, src_len, emb_dim)
        # Use mean-pooled encoder output as initial hidden/cell for decoder
        h0 = enc_out.mean(dim=1).unsqueeze(0)  # (1, B, emb_dim)
        c0 = torch.zeros_like(h0)
        outputs = []
        if trg is not None:
            trg_len = trg.size(1)
            input_token = trg[:, 0].unsqueeze(1)
            hidden, cell = h0, c0
            for t in range(1, trg_len):
                emb = self.embedding_out(input_token)
                output, (hidden, cell) = self.decoder_lstm(emb, (hidden, cell))
                prediction = self.fc_out(output.squeeze(1))
                outputs.append(prediction.unsqueeze(1))
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                input_token = trg[:, t].unsqueeze(1) if (teacher_force and t < trg_len) else prediction.argmax(1, keepdim=True)
            outputs = torch.cat(outputs, dim=1)
            return outputs
        else:
            input_token = torch.full((batch_size, 1), start_token_idx, dtype=torch.long, device=src.device)
            hidden, cell = h0, c0
            for _ in range(max_len):
                emb = self.embedding_out(input_token)
                output, (hidden, cell) = self.decoder_lstm(emb, (hidden, cell))
                prediction = self.fc_out(output.squeeze(1))
                outputs.append(prediction.unsqueeze(1))
                input_token = prediction.argmax(1, keepdim=True)
            outputs = torch.cat(outputs, dim=1)
            return outputs

# ========== Utility: Instantiate model ==========
if __name__ == "__main__":
    vocab_in_size = 200
    vocab_out_size = 100
    pad_idx_in = 0
    pad_idx_out = 0

    print("Testing BiLSTM model:")
    model1 = Seq2SeqSLM_BiLSTM(vocab_in_size, vocab_out_size, pad_idx_in=pad_idx_in, pad_idx_out=pad_idx_out)
    src = torch.randint(0, vocab_in_size, (4, 28))
    trg = torch.randint(0, vocab_out_size, (4, 16))
    out1 = model1(src, trg)
    print("BiLSTM Output shape:", out1.shape)

    print("Testing Transformer Encoder model:")
    model2 = Seq2SeqSLM_Transformer(vocab_in_size, vocab_out_size, pad_idx_in=pad_idx_in, pad_idx_out=pad_idx_out)
    out2 = model2(src, trg)
    print("Transformer Output shape:", out2.shape)
