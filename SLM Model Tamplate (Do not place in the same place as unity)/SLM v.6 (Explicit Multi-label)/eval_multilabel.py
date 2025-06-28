import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from slm_multilabel_model import SLM_Multilabel_BiLSTM, SLM_Multilabel_Transformer

# ========== CONFIG ==========
MODEL_PATH = "slm_multilabel.pt"
TOKENIZER_PATH = "tokenizer_multilabel.pkl"
ACTIONS_PATH = "all_actions.txt"
DATA_PATH = "sumobot_multilabel_dataset.csv"
LABEL_NPY = "sumobot_multilabel_labels.npy"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN_IN = 32
THRESHOLD = 0.5
USE_TRANSFORMER = False   
EVAL_SAMPLES = 200    

# ========== LOAD TOKENIZER & ACTIONS ==========
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
token2idx_in = tokenizer["token2idx_in"]
PAD_IDX_IN = token2idx_in["<PAD>"]

with open(ACTIONS_PATH, "r") as f:
    ALL_ACTIONS = [line.strip() for line in f.readlines()]
NUM_ACTIONS = len(ALL_ACTIONS)

# ========== LOAD DATA ==========
df = pd.read_csv(DATA_PATH)
labels = np.load(LABEL_NPY)
if EVAL_SAMPLES is not None and EVAL_SAMPLES < len(df):
    df_sampled = df.sample(n=EVAL_SAMPLES, random_state=42)
    labels_sampled = labels[df_sampled.index]
    df = df_sampled.reset_index(drop=True)
    labels = labels_sampled

# ========== LOAD MODEL ==========
if USE_TRANSFORMER:
    model = SLM_Multilabel_Transformer(
        vocab_in_size=len(token2idx_in), num_actions=NUM_ACTIONS, pad_idx_in=PAD_IDX_IN
    ).to(DEVICE)
else:
    model = SLM_Multilabel_BiLSTM(
        vocab_in_size=len(token2idx_in), num_actions=NUM_ACTIONS, pad_idx_in=PAD_IDX_IN
    ).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ========== TOKENIZER UTILS ==========
def tokenize_context(context_str):
    return context_str.lower().split()

def encode_tokens(seq, token2idx, unk_token="<UNK>", maxlen=None):
    ids = [token2idx.get(tok, token2idx.get(unk_token, 1)) for tok in seq]
    if maxlen:
        pad_idx = token2idx.get("<PAD>", 0)
        ids = ids[:maxlen] + [pad_idx] * (maxlen - len(ids)) if len(ids) < maxlen else ids[:maxlen]
    return ids

# ========== EVALUATIO  ==========
all_targets = []
all_preds = []

for i, row in df.iterrows():
    context_str = str(row["context_input"])
    target = labels[i]
    tokens_in = tokenize_context(context_str)
    ids_in = encode_tokens(tokens_in, token2idx_in, "<UNK>", MAX_LEN_IN)
    src = torch.tensor([ids_in], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits = model(src)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        pred_vec = (probs > THRESHOLD).astype(int)
    all_targets.append(target)
    all_preds.append(pred_vec)

all_targets = np.vstack(all_targets)
all_preds = np.vstack(all_preds)

# ==== METRICS ====
f1 = f1_score(all_targets, all_preds, average="micro", zero_division=0)
prec = precision_score(all_targets, all_preds, average="micro", zero_division=0)
recall = recall_score(all_targets, all_preds, average="micro", zero_division=0)

print("\n===== EVALUATION SUMMARY =====")
print(f"F1-score (micro):    {f1:.3f}")
print(f"Precision (micro):   {prec:.3f}")
print(f"Recall (micro):      {recall:.3f}")

# ==== Per-action report ====
print("\n=== PER-ACTION REPORT ===")
print(classification_report(all_targets, all_preds, target_names=ALL_ACTIONS, zero_division=0))

print("\n=== SAMPLE PREDICT ===")
for idx in range(5):
    true = [ALL_ACTIONS[i] for i, v in enumerate(all_targets[idx]) if v]
    pred = [ALL_ACTIONS[i] for i, v in enumerate(all_preds[idx]) if v]
    print(f"[{idx}]")
    print("Target :", true)
    print("Predict:", pred)
    print("===")
