import pandas as pd
import json
import pickle
from collections import Counter
import sys
import os

# ===================== CONFIG ====================
DATA_PATH = "sumobot_slm_merged.csv"  
TOKENIZER_OUT = "tokenizer_seq2seq.pkl"
VOCAB_JSON = "vocab_seq2seq.json"

print(f"=== [INFO] Current working directory: {os.getcwd()}")
print(f"=== [INFO] Looking for data file: {DATA_PATH}")

# ===================== Tokenization Functions =====================
def tokenize_strategy(strategy_str):
    tokens = []
    for tok in strategy_str.replace('+', ' + ').split():
        tokens.append(tok.strip())
    return ["<BOS>"] + tokens + ["<EOS>"]

def tokenize_instruction(instr_str):
    return instr_str.lower().split()

# ===================== Read Data & Tokenize =====================
try:
    df = pd.read_csv(DATA_PATH)
    print(f"=== [INFO] Loaded dataframe with shape: {df.shape}")
except Exception as e:
    print(f"=== [ERROR] Failed to load dataset: {e}")
    sys.exit(1)

input_tokenized = []
output_tokenized = []

print("=== [INFO] Tokenizing input and output...")
for i, row in df.iterrows():
    try:
        instr = str(row['situation'])
        strat = str(row['strategy'])
        input_tokenized.append(tokenize_instruction(instr))
        output_tokenized.append(tokenize_strategy(strat))
    except Exception as e:
        print(f"=== [WARN] Tokenization error at row {i}: {e}")
print(f"=== [INFO] Finished tokenizing {len(input_tokenized)} samples.")

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

SPECIAL_TOKENS_OUT = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
SPECIAL_TOKENS_IN = ["<PAD>", "<UNK>"]

print("=== [INFO] Building vocab for output (strategy)...")
vocab_out, token2idx_out, idx2token_out = build_vocab(output_tokenized, special_tokens=SPECIAL_TOKENS_OUT)
print(f"=== [INFO] Output vocab size: {len(vocab_out)}")

print("=== [INFO] Building vocab for input (situation)...")
vocab_in, token2idx_in, idx2token_in = build_vocab(input_tokenized, special_tokens=SPECIAL_TOKENS_IN)
print(f"=== [INFO] Input vocab size: {len(vocab_in)}")

# ===================== Save Tokenizer/Vocab =====================
tokenizer_data = {
    "vocab_in": vocab_in,
    "token2idx_in": token2idx_in,
    "idx2token_in": idx2token_in,
    "vocab_out": vocab_out,
    "token2idx_out": token2idx_out,
    "idx2token_out": idx2token_out,
}

try:
    with open(TOKENIZER_OUT, "wb") as f:
        pickle.dump(tokenizer_data, f)
    print(f"=== [INFO] Tokenizer saved to {TOKENIZER_OUT}")
except Exception as e:
    print(f"=== [ERROR] Failed to save {TOKENIZER_OUT}: {e}")

try:
    with open(VOCAB_JSON, "w") as f:
        json.dump({
            "vocab_in": vocab_in,
            "vocab_out": vocab_out
        }, f, indent=2)
    print(f"=== [INFO] Vocab saved to {VOCAB_JSON}")
except Exception as e:
    print(f"=== [ERROR] Failed to save {VOCAB_JSON}: {e}")

# ===================== Example Encode/Decode =====================
def encode_tokens(seq, token2idx, unk_token="<UNK>", maxlen=None):
    ids = [token2idx.get(tok, token2idx.get(unk_token, 1)) for tok in seq]
    if maxlen:
        pad_idx = token2idx.get("<PAD>", 0)
        ids = ids[:maxlen] + [pad_idx]*(maxlen-len(ids)) if len(ids)<maxlen else ids[:maxlen]
    return ids

def decode_ids(ids, idx2token):
    return [idx2token.get(i, "<UNK>") for i in ids]

print("=== [INFO] Sample encode/decode:")
if input_tokenized:
    sample_in = input_tokenized[0]
    sample_out = output_tokenized[0]
    print("Sample input tokens:", sample_in)
    print("Sample input ids:", encode_tokens(sample_in, token2idx_in, "<UNK>"))
    print("Sample output tokens:", sample_out)
    print("Sample output ids:", encode_tokens(sample_out, token2idx_out, "<UNK>"))
print("=== [DONE] tokenizer_seq2seq.py finished ===")
