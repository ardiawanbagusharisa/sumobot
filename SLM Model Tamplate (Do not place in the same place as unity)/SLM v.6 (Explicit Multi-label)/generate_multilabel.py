import torch
import pickle
import numpy as np
from slm_multilabel_model import SLM_Multilabel_BiLSTM, SLM_Multilabel_Transformer

# ========== CONFIG ==========
MODEL_PATH = "slm_multilabel.pt"
TOKENIZER_PATH = "tokenizer_multilabel.pkl"
ACTIONS_PATH = "all_actions.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN_IN = 32
THRESHOLD = 0.5
USE_TRANSFORMER = False 

# ========== LOAD TOKENIZER & ACTIONS ==========
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
token2idx_in = tokenizer["token2idx_in"]
idx2token_in = tokenizer["idx2token_in"]
PAD_IDX_IN = token2idx_in["<PAD>"]

with open(ACTIONS_PATH, "r") as f:
    ALL_ACTIONS = [line.strip() for line in f.readlines()]
NUM_ACTIONS = len(ALL_ACTIONS)

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

def decode_ids(ids, idx2token):
    return [idx2token.get(i, "<UNK>") for i in ids]

# ========== INFERENCE FUNCTION ==========
def generate_multilabel(context_str, threshold=THRESHOLD, print_logits=False):
    tokens_in = tokenize_context(context_str)
    ids_in = encode_tokens(tokens_in, token2idx_in, "<UNK>", MAX_LEN_IN)
    src = torch.tensor([ids_in], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits = model(src)  # (1, NUM_ACTIONS)
        probs = torch.sigmoid(logits)[0].cpu().numpy()  # (NUM_ACTIONS,)
        if print_logits:
            print("Aksi & Probabilitas:")
            for action, p in zip(ALL_ACTIONS, probs):
                print(f"{action:20s}: {p:.3f}")
        pred_actions = [ALL_ACTIONS[i] for i, p in enumerate(probs) if p > threshold]
    strategy = "+".join(pred_actions)
    return strategy, pred_actions, probs

# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    context = "Enemy is 1.2 meters ahead angle 90 dash ready [CTX] Enemy is 1.0 meters ahead angle 80 dash ready"
    print("Context input:", context)
    strategy, actions, probs = generate_multilabel(context, threshold=THRESHOLD, print_logits=True)
    print("\nPredicted actions:", actions)
    print("Predicted strategy string:", strategy)
