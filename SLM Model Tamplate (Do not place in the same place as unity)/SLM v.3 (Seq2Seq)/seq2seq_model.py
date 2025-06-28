import torch
import torch.nn as nn

class Seq2SeqSLM(nn.Module):
    def __init__(self, vocab_in_size, vocab_out_size, emb_dim=128, hidden_dim=256, num_layers=1, pad_idx_in=0, pad_idx_out=0):
        super().__init__()
        # Encoder for input instructions
        self.embedding_in = nn.Embedding(vocab_in_size, emb_dim, padding_idx=pad_idx_in)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True)

        # Decoder for output strategy tokens
        self.embedding_out = nn.Embedding(vocab_out_size, emb_dim, padding_idx=pad_idx_out)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_out_size)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5, max_len=16, start_token_idx=1):
        batch_size = src.size(0)

        # ========== ENCODER ==========
        embedded_src = self.embedding_in(src)
        _, (h, c) = self.encoder(embedded_src)

        # ========== DECODER ==========
        outputs = []
        if trg is not None:
            trg_len = trg.size(1)
            input_token = trg[:, 0].unsqueeze(1)  
            hidden, cell = h, c
            for t in range(1, trg_len):
                embedded_trg = self.embedding_out(input_token) 
                output, (hidden, cell) = self.decoder(embedded_trg, (hidden, cell)) 
                prediction = self.fc_out(output.squeeze(1))      
                outputs.append(prediction.unsqueeze(1))

                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                input_token = trg[:, t].unsqueeze(1) if (teacher_force and t < trg_len) else prediction.argmax(1, keepdim=True)
            outputs = torch.cat(outputs, dim=1)  
            return outputs
        else:
            # Inference (auto-regressive, until <EOS> or max_len)
            input_token = torch.full((batch_size, 1), start_token_idx, dtype=torch.long, device=src.device) 
            hidden, cell = h, c
            for _ in range(max_len):
                embedded_trg = self.embedding_out(input_token)
                output, (hidden, cell) = self.decoder(embedded_trg, (hidden, cell))
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
    model = Seq2SeqSLM(vocab_in_size, vocab_out_size, pad_idx_in=pad_idx_in, pad_idx_out=pad_idx_out)
    src = torch.randint(0, vocab_in_size, (4, 14))      
    trg = torch.randint(0, vocab_out_size, (4, 10))      
    output = model(src, trg)  
    print("Output shape:", output.shape)
