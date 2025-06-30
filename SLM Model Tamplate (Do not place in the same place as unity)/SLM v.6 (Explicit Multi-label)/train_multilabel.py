import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import numpy as np
from slm_multilabel_model import SLM_Multilabel_BiLSTM, SLM_Multilabel_Transformer
from sklearn.metrics import f1_score, precision_score, recall_score

# ========== CONFIG ==========
DATA_PATH = "sumobot_multilabel_dataset.csv"
LABEL_NPY = "sumobot_multilabel_labels.npy"
TOKENIZER_PATH = "tokenizer_multilabel.pkl"
ACTIONS_PATH = "all_actions.txt"
MODEL_PATH = "slm_multilabel.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 30
MAX_LEN_IN = 32

USE_TRANSFORMER = False
EMB_DIM = 128
HIDDEN_DIM = 256
NHEAD = 4
NUM_LAYERS = 2

# ========== LOAD TOKENIZER & ACTIONS ==========
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
token2idx_in = tokenizer["token2idx_in"]
PAD_IDX_IN = token2idx_in["<PAD>"]

with open(ACTIONS_PATH, "r") as f:
    ALL_ACTIONS = [line.strip() for line in f.readlines()]
NUM_ACTIONS = len(ALL_ACTIONS)

# ========== DATASET & DATALOADER ==========
class MultiLabelDataset(Dataset):
    def __init__(self, csv_path, label_npy, max_len=MAX_LEN_IN):
        self.df = pd.read_csv(csv_path)
        self.labels = np.load(label_npy)
        self.input_tokenized = [str(s).lower().split() for s in self.df["context_input"].astype(str)]
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        inp = self.input_tokenized[idx]
        # Tokenize + pad
        inp_ids = [token2idx_in.get(tok, token2idx_in["<UNK>"]) for tok in inp][:self.max_len]
        inp_ids += [PAD_IDX_IN] * (self.max_len - len(inp_ids))
        label = self.labels[idx].astype(np.float32)
        return torch.tensor(inp_ids), torch.tensor(label)

dataset = MultiLabelDataset(DATA_PATH, LABEL_NPY)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ========== MODEL ==========
if USE_TRANSFORMER:
    model = SLM_Multilabel_Transformer(
        vocab_in_size=len(token2idx_in), num_actions=NUM_ACTIONS, emb_dim=EMB_DIM, nhead=NHEAD, num_layers=NUM_LAYERS, pad_idx_in=PAD_IDX_IN
    ).to(DEVICE)
else:
    model = SLM_Multilabel_BiLSTM(
        vocab_in_size=len(token2idx_in), num_actions=NUM_ACTIONS, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, num_layers=1, pad_idx_in=PAD_IDX_IN
    ).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ========== TRAINING ==========
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    all_targets = []
    all_preds = []
    for batch in loader:
        inp, target = batch
        inp, target = inp.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        logits = model(inp)  # (B, NUM_ACTIONS)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # For metrics (batch-wise, sigmoid then >0.5)
        probs = torch.sigmoid(logits).cpu().detach().numpy()
        preds = (probs > 0.5).astype(int)
        all_targets.append(target.cpu().numpy())
        all_preds.append(preds)
    # Epoch metrics
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    f1 = f1_score(all_targets, all_preds, average="micro", zero_division=0)
    prec = precision_score(all_targets, all_preds, average="micro", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="micro", zero_division=0)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f} | F1: {f1:.3f} | Prec: {prec:.3f} | Rec: {recall:.3f}")

# ========== SAVE MODEL ==========
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
