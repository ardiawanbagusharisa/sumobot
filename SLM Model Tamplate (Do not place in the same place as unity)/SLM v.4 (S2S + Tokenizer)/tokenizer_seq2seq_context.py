import pandas as pd
import json
import pickle
from collections import Counter

# ===================== CONFIG =====================
DATA_PATH = "sumobot_context_dataset.csv"  
TOKENIZER_OUT = "tokenizer_seq2seq_context.pkl"
VOCAB_JSON = "vocab_seq2seq_context.json"

# ===================== Tokenization Functions =====================

def tokenize_context_input(context_str):
    return context_str.lower().split()

def tokenize_strategy(strategy_str):
    tokens = []
    for tok in strategy_str.replace('+', ' + ').split():
        tokens.append(tok.strip())
    return ["<BOS>"] + tokens + ["<EOS>"]

# ===================== Read Data & Tokenize =====================
df = pd.read_csv(DATA_PATH)
input_tokenized = [tokenize_context_input(str(s)) for s in df["context_input"].astype(str)]
output_tokenized = [tokenize_strategy(str(s)) for s in df["strategy"].astype(str)]

# ===================== Build Vocab =====================

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

# Input vocab (context)
SPECIAL_TOKENS_IN = ["<PAD>", "<UNK>", "[ctx]"]
vocab_in, token2idx_in, idx2token_in = build_vocab(input_tokenized, special_tokens=SPECIAL_TOKENS_IN)

# Output vocab (strategy)
SPECIAL_TOKENS_OUT = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
vocab_out, token2idx_out, idx2token_out = build_vocab(output_tokenized, special_tokens=SPECIAL_TOKENS_OUT)

print(f"Input vocab size: {len(vocab_in)}")
print(f"Output vocab size: {len(vocab_out)}")

# ===================== Save Tokenizer/Vocab =====================

tokenizer_data = {
    "vocab_in": vocab_in,
    "token2idx_in": token2idx_in,
    "idx2token_in": idx2token_in,
    "vocab_out": vocab_out,
    "token2idx_out": token2idx_out,
    "idx2token_out": idx2token_out,
}

with open(TOKENIZER_OUT, "wb") as f:
    pickle.dump(tokenizer_data, f)

with open(VOCAB_JSON, "w") as f:
    json.dump({
        "vocab_in": vocab_in,
        "vocab_out": vocab_out
    }, f, indent=2)

print(f"Tokenizer and vocab saved to {TOKENIZER_OUT} and {VOCAB_JSON}")

# ===================== Example Encode/Decode =====================
def encode_tokens(seq, token2idx, unk_token="<UNK>", maxlen=None):
    ids = [token2idx.get(tok, token2idx.get(unk_token, 1)) for tok in seq]
    if maxlen:
        pad_idx = token2idx.get("<PAD>", 0)
        ids = ids[:maxlen] + [pad_idx] * (maxlen - len(ids)) if len(ids) < maxlen else ids[:maxlen]
    return ids

def decode_ids(ids, idx2token):
    return [idx2token.get(i, "<UNK>") for i in ids]

# Test sample
sample_in = input_tokenized[0]
sample_out = output_tokenized[0]
print("Sample context input:", sample_in)
print("Encoded:", encode_tokens(sample_in, token2idx_in, "<UNK>"))
print("Sample output:", sample_out)
print("Encoded:", encode_tokens(sample_out, token2idx_out, "<UNK>"))
