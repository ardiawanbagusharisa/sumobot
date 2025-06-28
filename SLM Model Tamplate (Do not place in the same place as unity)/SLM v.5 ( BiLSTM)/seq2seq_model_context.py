import torch
import torch.nn as nn

def top_k_sampling(logits, k=3, temperature=1.0):
    """
    logits: (batch, vocab_size)
    Return: (batch, 1) sampled token idx
    """
    # Apply temperature
    probs = torch.softmax(logits / temperature, dim=-1)
    topk_probs, topk_idx = torch.topk(probs, k, dim=-1)
    # Normalize
    topk_probs = topk_probs / torch.sum(topk_probs, dim=-1, keepdim=True)
    next_token = torch.multinomial(topk_probs, 1)
    sampled = topk_idx.gather(-1, next_token)
    return sampled

class Seq2SeqSLM_BiLSTM(nn.Module):
    def __init__(self, vocab_in_size, vocab_out_size, emb_dim=128, hidden_dim=256, num_layers=1, pad_idx_in=0, pad_idx_out=0, 
                 sampling_k=1, sampling_temp=1.0):
        super().__init__()
        # ----- Encoder: BiLSTM -----
        self.embedding_in = nn.Embedding(vocab_in_size, emb_dim, padding_idx=pad_idx_in)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # ----- Decoder: LSTM -----
        self.embedding_out = nn.Embedding(vocab_out_size, emb_dim, padding_idx=pad_idx_out)
        self.decoder = nn.LSTM(emb_dim, hidden_dim*2, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim*2, vocab_out_size)
        # Sampling params
        self.sampling_k = sampling_k
        self.sampling_temp = sampling_temp

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5, max_len=16, start_token_idx=1, do_sampling=False):
        # src: (B, src_len), trg: (B, trg_len)
        batch_size = src.size(0)
        # ----- Encoder -----
        embedded_src = self.embedding_in(src)
        _, (h, c) = self.encoder(embedded_src)
        # Merge fwd & bwd last layer
        h_cat = torch.cat([h[-2], h[-1]], dim=1).unsqueeze(0)
        c_cat = torch.cat([c[-2], c[-1]], dim=1).unsqueeze(0)
        outputs = []
        if trg is not None:
            # Training, teacher forcing
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
            # Inference/generation (auto-regressive)
            input_token = torch.full((batch_size, 1), start_token_idx, dtype=torch.long, device=src.device)
            hidden, cell = h_cat, c_cat
            for _ in range(max_len):
                emb = self.embedding_out(input_token)
                output, (hidden, cell) = self.decoder(emb, (hidden, cell))
                prediction = self.fc_out(output.squeeze(1))  # (B, vocab_out)
                outputs.append(prediction.unsqueeze(1))
                # --- Sampling or greedy ---
                if do_sampling and self.sampling_k > 1:
                    input_token = top_k_sampling(prediction, k=self.sampling_k, temperature=self.sampling_temp)
                else:
                    input_token = prediction.argmax(1, keepdim=True)
            outputs = torch.cat(outputs, dim=1)
            return outputs

class Seq2SeqSLM_Transformer(nn.Module):
    def __init__(self, vocab_in_size, vocab_out_size, emb_dim=128, nhead=4, num_layers=2, pad_idx_in=0, pad_idx_out=0,
                 sampling_k=1, sampling_temp=1.0):
        super().__init__()
        # ----- Encoder: Transformer -----
        self.embedding_in = nn.Embedding(vocab_in_size, emb_dim, padding_idx=pad_idx_in)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=emb_dim*2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # ----- Decoder: LSTM -----
        self.embedding_out = nn.Embedding(vocab_out_size, emb_dim, padding_idx=pad_idx_out)
        self.decoder_lstm = nn.LSTM(emb_dim, emb_dim, num_layers=1, batch_first=True)
        self.fc_out = nn.Linear(emb_dim, vocab_out_size)
        # Sampling params
        self.sampling_k = sampling_k
        self.sampling_temp = sampling_temp

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5, max_len=16, start_token_idx=1, do_sampling=False):
        batch_size = src.size(0)
        # ----- Encoder -----
        embedded_src = self.embedding_in(src)
        src_mask = None
        enc_out = self.encoder(embedded_src, mask=src_mask)
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
                # --- Sampling or greedy ---
                if do_sampling and self.sampling_k > 1:
                    input_token = top_k_sampling(prediction, k=self.sampling_k, temperature=self.sampling_temp)
                else:
                    input_token = prediction.argmax(1, keepdim=True)
            outputs = torch.cat(outputs, dim=1)
            return outputs

# ========== Utility: Instantiate model ==========
if __name__ == "__main__":
    vocab_in_size = 200
    vocab_out_size = 100
    pad_idx_in = 0
    pad_idx_out = 0

    print("Testing BiLSTM model (sampling):")
    model1 = Seq2SeqSLM_BiLSTM(vocab_in_size, vocab_out_size, pad_idx_in=pad_idx_in, pad_idx_out=pad_idx_out,
                               sampling_k=3, sampling_temp=1.0)
    src = torch.randint(0, vocab_in_size, (4, 28))
    trg = torch.randint(0, vocab_out_size, (4, 16))
    out1 = model1(src, trg)
    print("BiLSTM Output shape:", out1.shape)
    out1_inf = model1(src, trg=None, do_sampling=True)
    print("BiLSTM Output shape (inference, sampling):", out1_inf.shape)

    print("Testing Transformer Encoder model (sampling):")
    model2 = Seq2SeqSLM_Transformer(vocab_in_size, vocab_out_size, pad_idx_in=pad_idx_in, pad_idx_out=pad_idx_out,
                                    sampling_k=3, sampling_temp=1.0)
    out2 = model2(src, trg)
    print("Transformer Output shape:", out2.shape)
    out2_inf = model2(src, trg=None, do_sampling=True)
    print("Transformer Output shape (inference, sampling):", out2_inf.shape)
