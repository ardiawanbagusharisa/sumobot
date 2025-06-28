import torch
from flask import Flask, request, jsonify
import pickle
from seq2seq_model import Seq2SeqSLM

# ===== CONFIG =====
MODEL_PATH = "seq2seq_slm.pt"
TOKENIZER_PATH = "tokenizer_seq2seq.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN_IN = 24
MAX_LEN_OUT = 14

# ===== LOAD TOKENIZER & MODEL =====
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

app = Flask(__name__)

feature_columns = [
    "enemy_distance", "enemy_angle", "edge_distance", "center_distance",
    "enemy_stuck", "enemy_behind", "skill_ready", "dash_ready"
]

def features_to_text(data):
    return (f"Enemy is {data['enemy_distance']} meters ahead, "
            f"angle {data['enemy_angle']} degrees, "
            f"edge distance {data['edge_distance']}, "
            f"center distance {data['center_distance']}, "
            f"enemy stuck: {bool(data['enemy_stuck'])}, "
            f"enemy behind: {bool(data['enemy_behind'])}, "
            f"skill ready: {bool(data['skill_ready'])}, "
            f"dash ready: {bool(data['dash_ready'])}")

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
    tokens_in = tokenize_instruction(instr_text)
    ids_in = encode_tokens(tokens_in, token2idx_in, "<UNK>", MAX_LEN_IN)
    src = torch.tensor([ids_in], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        outputs = model(src, trg=None, teacher_forcing_ratio=0.0, max_len=max_gen_len, start_token_idx=BOS_IDX)
        pred_ids = outputs.argmax(-1)[0].tolist()
        tokens = []
        for tok_id in pred_ids:
            tok = idx2token_out.get(tok_id, "<UNK>")
            if tok == "<EOS>":
                break
            tokens.append(tok)
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
    return strategy

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if all(k in data for k in feature_columns):
            instr_text = features_to_text(data)
        else:
            instr_text = data.get("situation", "")
        if not instr_text:
            return jsonify({"error": "No instruction provided", "strategy": "stay"}), 400
        strategy = generate_strategy(instr_text)
        return jsonify({"strategy": strategy})
    except Exception as e:
        return jsonify({"error": str(e), "strategy": "stay"}), 500

if __name__ == '__main__':
    print("Seq2Seq SLM API running at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
