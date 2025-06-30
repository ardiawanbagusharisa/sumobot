import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
from seq2seq_model import Seq2SeqSLM

# =================== CONFIG ===================
DATA_PATH = "sumobot_slm_merged.csv"
TOKENIZER_PATH = "tokenizer_seq2seq.pkl"
MODEL_PATH = "seq2seq_slm.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 30
TEACHER_FORCING = 0.7
MAX_LEN_IN = 24    
MAX_LEN_OUT = 14  

# =================== Load Tokenizer ===================
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

token2idx_in = tokenizer["token2idx_in"]
idx2token_in = tokenizer["idx2token_in"]
token2idx_out = tokenizer["token2idx_out"]
idx2token_out = tokenizer["idx2token_out"]

PAD_IDX_IN = token2idx_in["<PAD>"]
PAD_IDX_OUT = token2idx_out["<PAD>"]
BOS_IDX = token2idx_out["<BOS>"]
EOS_IDX = token2idx_out["<EOS>"]

vocab_in_size = len(token2idx_in)
vocab_out_size = len(token2idx_out)

# =================== Dataset & Loader ===================
class SumoSeq2SeqDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.input_tokenized = [s.lower().split() for s in self.df["situation"].astype(str)]
        self.output_tokenized = []
        for s in self.df["strategy"].astype(str):
            toks = []
            for t in s.replace("+", " + ").split():
                toks.append(t.strip())
            self.output_tokenized.append(["<BOS>"] + toks + ["<EOS>"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Encode and pad
        inp = self.input_tokenized[idx]
        out = self.output_tokenized[idx]
        # Padding
        inp_ids = [token2idx_in.get(tok, token2idx_in["<UNK>"]) for tok in inp][:MAX_LEN_IN]
        out_ids = [token2idx_out.get(tok, token2idx_out["<UNK>"]) for tok in out][:MAX_LEN_OUT]
        # Pad
        inp_ids += [PAD_IDX_IN] * (MAX_LEN_IN - len(inp_ids))
        out_ids += [PAD_IDX_OUT] * (MAX_LEN_OUT - len(out_ids))
        return torch.tensor(inp_ids), torch.tensor(out_ids)

dataset = SumoSeq2SeqDataset(DATA_PATH)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =================== Model, Loss, Optim ===================
model = Seq2SeqSLM(
    vocab_in_size, vocab_out_size, emb_dim=128, hidden_dim=256, 
    pad_idx_in=PAD_IDX_IN, pad_idx_out=PAD_IDX_OUT
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX_OUT)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =================== Training Loop ===================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inp, out = batch
        inp, out = inp.to(DEVICE), out.to(DEVICE)
        optimizer.zero_grad()
        # Remove last token for decoder input; targets are next tokens
        dec_in = out[:, :-1]     
        dec_target = out[:, 1:]  
        logits = model(inp, dec_in, teacher_forcing_ratio=TEACHER_FORCING, max_len=dec_target.size(1), start_token_idx=BOS_IDX)

        if logits.size(1) > dec_target.size(1):
            logits = logits[:, :dec_target.size(1), :]
        elif logits.size(1) < dec_target.size(1):
            dec_target = dec_target[:, :logits.size(1)]

        loss = criterion(logits.reshape(-1, vocab_out_size), dec_target.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

# =================== Save Model ===================
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
