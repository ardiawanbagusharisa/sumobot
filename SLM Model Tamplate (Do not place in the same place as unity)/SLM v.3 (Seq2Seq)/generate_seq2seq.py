import torch
import pickle
from seq2seq_model import Seq2SeqSLM

# =============== CONFIG =================
MODEL_PATH = "seq2seq_slm.pt"
TOKENIZER_PATH = "tokenizer_seq2seq.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN_IN = 24
MAX_LEN_OUT = 14

# =============== LOAD TOKENIZER & MODEL =================
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
token2idx_in = tokenizer["token2idx_in"]
idx2token_in = tokenizer["idx2token_in"]
token2idx_out = tokenizer["token2idx_out"]
idx2token_out = tokenizer["idx2token_out"]
PAD_IDX_IN = token2idx_in["<PAD>"]
BOS_IDX = token2idx_out["<BOS>"]
EOS_IDX = token2idx_out["<EOS>"]

vocab_in_size = len(token2idx_in)
vocab_out_size = len(token2idx_out)

model = Seq2SeqSLM(
    vocab_in_size, vocab_out_size, emb_dim=128, hidden_dim=256,
    pad_idx_in=PAD_IDX_IN, pad_idx_out=token2idx_out["<PAD>"]
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =============== INFERENCE FUNCTION =================
def tokenize_instruction(instr):
    return instr.lower().split()

def encode_tokens(seq, token2idx, unk_token="<UNK>", maxlen=None):
    ids = [token2idx.get(tok, token2idx.get(unk_token, 1)) for tok in seq]
    if maxlen:
        pad_idx = token2idx.get("<PAD>", 0)
        ids = ids[:maxlen] + [pad_idx] * (maxlen - len(ids)) if len(ids) < maxlen else ids[:maxlen]
    return ids

def decode_ids(ids, idx2token):
    return [idx2token.get(i, "<UNK>") for i in ids]

def generate_strategy(instr_text, max_gen_len=MAX_LEN_OUT):
    # Tokenize and encode input
    tokens_in = tokenize_instruction(instr_text)
    ids_in = encode_tokens(tokens_in, token2idx_in, "<UNK>", MAX_LEN_IN)
    src = torch.tensor([ids_in], dtype=torch.long).to(DEVICE)

    # Forward pass (auto-regressive generation)
    with torch.no_grad():
        outputs = model(src, trg=None, teacher_forcing_ratio=0.0, max_len=max_gen_len, start_token_idx=BOS_IDX)
        # outputs: logits per token
        pred_ids = outputs.argmax(-1)[0].tolist()  # (max_len, )
        # Stop at first EOS (exclude BOS)
        tokens = []
        for tok_id in pred_ids:
            tok = idx2token_out.get(tok_id, "<UNK>")
            if tok == "<EOS>":
                break
            tokens.append(tok)
        # Remove initial BOS if present
        if tokens and tokens[0] == "<BOS>":
            tokens = tokens[1:]

    strategy = ""
    for tok in tokens:
        if tok == "+":
            strategy += "+"
        else:
            if strategy and not strategy.endswith("+"):
                strategy += "+"
            strategy += tok
    strategy = strategy.replace("++", "+").strip("+")
    return strategy, tokens

# =============== TEST EXAMPLE =================
if __name__ == "__main__":
    instr = "Enemy is 1.2 meters ahead, angle 90 degrees, edge distance 0.6, center distance 2.5, enemy stuck: 0, enemy behind: 0, skill ready: 1, dash ready: 1"
    strategy, tokens = generate_strategy(instr)
    print("Instruction:", instr)
    print("Generated tokens:", tokens)
    print("Generated strategy string:", strategy)
