import pickle
import json

# ===== CONFIG =====
TOKENIZER_PATH = "tokenizer_seq2seq.pkl"
VOCAB_JSON = "vocab_seq2seq.json"

# ===== LOAD TOKENIZER =====
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

token2idx_in = tokenizer["token2idx_in"]
idx2token_in = tokenizer["idx2token_in"]
token2idx_out = tokenizer["token2idx_out"]
idx2token_out = tokenizer["idx2token_out"]

# ===== FUNCTIONS =====

def preview_vocab(input_or_output="input", show_n=30):
    if input_or_output == "input":
        vocab = list(token2idx_in.keys())
    else:
        vocab = list(token2idx_out.keys())
    print(f"Vocab ({input_or_output}): {len(vocab)} tokens")
    print(", ".join(vocab[:show_n]) + (" ..." if len(vocab) > show_n else ""))

def encode(seq, token2idx, unk_token="<UNK>", maxlen=None):
    ids = [token2idx.get(tok, token2idx.get(unk_token, 1)) for tok in seq]
    if maxlen:
        pad_idx = token2idx.get("<PAD>", 0)
        ids = ids[:maxlen] + [pad_idx] * (maxlen - len(ids)) if len(ids) < maxlen else ids[:maxlen]
    return ids

def decode(ids, idx2token):
    return [idx2token.get(i, "<UNK>") for i in ids]

def find_oov(tokens, token2idx):
    return [tok for tok in tokens if tok not in token2idx]

def update_vocab_from_new_data(new_tokens, token2idx, idx2token, special_tokens=None):
    cur_max = max(token2idx.values())
    added = 0
    for tok in new_tokens:
        if tok not in token2idx:
            cur_max += 1
            token2idx[tok] = cur_max
            idx2token[cur_max] = tok
            added += 1
    print(f"Added {added} new tokens.")
    return token2idx, idx2token

# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    preview_vocab("input", show_n=20)
    preview_vocab("output", show_n=20)

    sample_in = "enemy is 1.1 meters ahead angle 90 degrees".split()
    ids = encode(sample_in, token2idx_in)
    print("Encode input:", ids)
    print("Decode input:", decode(ids, idx2token_in))

    sample_out = ["<BOS>", "turn_left_90", "+", "accelerate", "<EOS>"]
    ids_out = encode(sample_out, token2idx_out)
    print("Encode output:", ids_out)
    print("Decode output:", decode(ids_out, idx2token_out))

    oov = find_oov(["enemy", "super_dash", "999"], token2idx_in)
    print("OOV tokens (input):", oov)

    new_tokens = ["retreat", "spin_attack"]
    updated_token2idx_out, updated_idx2token_out = update_vocab_from_new_data(new_tokens, token2idx_out, idx2token_out)

    with open("tokenizer_seq2seq_updated.pkl", "wb") as f:
        pickle.dump({
            "token2idx_in": token2idx_in,
            "idx2token_in": idx2token_in,
            "token2idx_out": updated_token2idx_out,
            "idx2token_out": updated_idx2token_out,
        }, f)
    print("Updated tokenizer saved as tokenizer_seq2seq_updated.pkl")
