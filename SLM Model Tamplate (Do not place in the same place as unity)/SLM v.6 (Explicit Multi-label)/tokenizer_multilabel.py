import pandas as pd
import pickle
import json
from collections import Counter

# ===== CONFIG =====
DATA_PATH = "sumobot_multilabel_dataset.csv" 
TOKENIZER_OUT = "tokenizer_multilabel.pkl"
VOCAB_JSON = "vocab_multilabel.json"

# ===== 1. Tokenization Function =====
def tokenize_context_input(context_str):
    # Simple word-level: split by space (including [CTX] as a token)
    return context_str.lower().split()

# ===== 2. Read Data & Tokenize =====
df = pd.read_csv(DATA_PATH)
input_tokenized = [tokenize_context_input(str(s)) for s in df["context_input"].astype(str)]

# ===== 3. Build Vocab =====
def build_vocab(tokenized_list, special_tokens=None, min_freq=1):
    counter = Counter(token for seq in tokenized_list for token in seq)
    tokens = [tok for tok, cnt in counter.items() if cnt >= min_freq]
    if special_tokens:
        vocab = special_tokens + [tok for tok in tokens if tok not in special_tokens]
    else:
        vocab = tokens
    token2idx = {tok: idx for idx, tok in enumerate(vocab)}
    idx2token = {idx: tok for tok, idx in token2idx.items()}
    return vocab, token2idx, idx2token

SPECIAL_TOKENS_IN = ["<PAD>", "<UNK>", "[ctx]"]
vocab_in, token2idx_in, idx2token_in = build_vocab(input_tokenized, special_tokens=SPECIAL_TOKENS_IN)

print(f"Input vocab size: {len(vocab_in)}")

# ===== 4. Save Tokenizer/Vocab =====
tokenizer_data = {
    "vocab_in": vocab_in,
    "token2idx_in": token2idx_in,
    "idx2token_in": idx2token_in,
}

with open(TOKENIZER_OUT, "wb") as f:
    pickle.dump(tokenizer_data, f)

with open(VOCAB_JSON, "w") as f:
    json.dump({
        "vocab_in": vocab_in
    }, f, indent=2)

print(f"Tokenizer and vocab saved to {TOKENIZER_OUT} and {VOCAB_JSON}")

# ===== 5. Example Encode/Decode =====
def encode_tokens(seq, token2idx, unk_token="<UNK>", maxlen=None):
    ids = [token2idx.get(tok, token2idx.get(unk_token, 1)) for tok in seq]
    if maxlen:
        pad_idx = token2idx.get("<PAD>", 0)
        ids = ids[:maxlen] + [pad_idx]*(maxlen - len(ids)) if len(ids) < maxlen else ids[:maxlen]
    return ids

def decode_ids(ids, idx2token):
    return [idx2token.get(i, "<UNK>") for i in ids]

sample_in = input_tokenized[0]
print("Sample context input:", sample_in)
print("Encoded:", encode_tokens(sample_in, token2idx_in, "<UNK>"))
