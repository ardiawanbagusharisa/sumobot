import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
from seq2seq_model_context import Seq2SeqSLM_BiLSTM, Seq2SeqSLM_Transformer

# =============== CONFIG ================
DATA_PATH = "sumobot_context_dataset.csv"
TOKENIZER_PATH = "tokenizer_seq2seq_context.pkl"
MODEL_PATH = "seq2seq_context.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 30
TEACHER_FORCING = 0.7
MAX_LEN_IN = 32   
MAX_LEN_OUT = 16    

# ======= MODEL CHOICE & SAMPLING =======
USE_TRANSFORMER = False  
SAMPLING_K = 3           
SAMPLING_TEMP = 1.0

# =============== LOAD TOKENIZER ================
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

# =============== DATASET & LOADER ================
class ContextSeq2SeqDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.input_tokenized = [s.lower().split() for s in self.df["context_input"].astype(str)]
        self.output_tokenized = []
        for s in self.df["strategy"].astype(str):
            toks = []
            for t in s.replace("+", " + ").split():
                toks.append(t.strip())
            self.output_tokenized.append(["<BOS>"] + toks + ["<EOS>"])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        inp = self.input_tokenized[idx]
        out = self.output_tokenized[idx]
        # Padding
        inp_ids = [token2idx_in.get(tok, token2idx_in["<UNK>"]) for tok in inp][:MAX_LEN_IN]
        out_ids = [token2idx_out.get(tok, token2idx_out["<UNK>"]) for tok in out][:MAX_LEN_OUT]
        inp_ids += [PAD_IDX_IN] * (MAX_LEN_IN - len(inp_ids))
        out_ids += [PAD_IDX_OUT] * (MAX_LEN_OUT - len(out_ids))
        return torch.tensor(inp_ids), torch.tensor(out_ids)

dataset = ContextSeq2SeqDataset(DATA_PATH)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =============== MODEL, LOSS, OPTIM ================
model_args = dict(
    vocab_in_size=vocab_in_size,
    vocab_out_size=vocab_out_size,
    pad_idx_in=PAD_IDX_IN,
    pad_idx_out=PAD_IDX_OUT,
    sampling_k=SAMPLING_K,
    sampling_temp=SAMPLING_TEMP
)
if USE_TRANSFORMER:
    model = Seq2SeqSLM_Transformer(
        emb_dim=128, nhead=4, num_layers=2, **model_args
    ).to(DEVICE)
else:
    model = Seq2SeqSLM_BiLSTM(
        emb_dim=128, hidden_dim=256, **model_args
    ).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX_OUT)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =============== TRAINING LOOP ================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inp, out = batch
        inp, out = inp.to(DEVICE), out.to(DEVICE)
        optimizer.zero_grad()
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

    # ======= SAMPLING EVALUATION =======
    if epoch % 5 == 0 or epoch == EPOCHS-1:
        model.eval()
        with torch.no_grad():
            test_batch = next(iter(train_loader))
            sample_in, _ = test_batch
            sample_in = sample_in.to(DEVICE)
            # Generate with top-k sampling
            gen_logits = model(sample_in, trg=None, do_sampling=True, max_len=MAX_LEN_OUT, start_token_idx=BOS_IDX)
            pred_ids = gen_logits.argmax(-1).cpu().tolist()  # (B, max_len)
            for b in range(min(2, BATCH_SIZE)):
                tokens = [idx2token_out.get(i, "<UNK>") for i in pred_ids[b]]
                print(f"Sample generated tokens: {' '.join(tokens)}")
            print("----")

# =============== SAVE MODEL ================
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
