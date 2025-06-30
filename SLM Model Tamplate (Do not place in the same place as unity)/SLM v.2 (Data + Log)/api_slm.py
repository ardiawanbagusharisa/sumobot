import torch
from flask import Flask, request, jsonify
import json
import pickle

from slm_model import SLMModel, SimpleTokenizer

feature_columns = [
    "enemy_distance", "enemy_angle", "edge_distance", "center_distance",
    "enemy_stuck", "enemy_behind", "skill_ready", "dash_ready"
]

app = Flask(__name__) 

# ====== 1. Load Model, Tokenizer, Mapping ======
with open("action2idx.json") as f:
    action2idx = json.load(f)
with open("idx2action.json") as f:
    idx2action = json.load(f)
with open("slm_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word2idx)
output_dim = len(action2idx)
emb_dim = 64
hidden_dim = 128

model = SLMModel(vocab_size, emb_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("slm_model.pt", map_location=torch.device("cpu")))
model.eval()

max_len = 20

def features_to_text(data):
    return (f"enemy position {data['enemy_distance']} meters ahead, "
            f"angle {data['enemy_angle']} degrees, "
            f"distance to edge {data['edge_distance']}, "
            f"distance to center {data['center_distance']}, "
            f"enemy stuck: {bool(data['enemy_stuck'])}, "
            f"enemy behind: {bool(data['enemy_behind'])}, "
            f"skill ready: {bool(data['skill_ready'])}, "
            f"dash ready: {bool(data['dash_ready'])}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if all(k in data for k in feature_columns):
            instruksi = features_to_text(data)
        else:
            instruksi = data.get("situation", "")
      
        ids = tokenizer.encode(instruksi, max_len)
        input_tensor = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(input_tensor)
            idx = int(logits.argmax(dim=1)[0])
            aksi = idx2action[str(idx)] if isinstance(idx2action, dict) else idx2action[idx]
        return jsonify({"strategy": aksi})
    except Exception as e:
        return jsonify({"error": str(e), "strategy": "stay"}), 500

if __name__ == '__main__':
    print("SLM API running at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
