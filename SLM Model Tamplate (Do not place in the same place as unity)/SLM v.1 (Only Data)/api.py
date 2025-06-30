import torch
import torch.nn.functional as F
from model import StrategyEmbeddingModel
import json
from flask import Flask, request, jsonify
import numpy as np

# ===== 1. Load Mapping dan Model =====
with open("action2idx.json") as f:
    action2idx = json.load(f)
with open("idx2action.json") as f:
    idx2action = json.load(f)

input_dim = 8
hidden_dim = 32
output_dim = len(action2idx)

model = StrategyEmbeddingModel(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
model.eval()

app = Flask(__name__)

feature_columns = [
    "enemy_distance", "enemy_angle", "edge_distance", "center_distance",
    "enemy_stuck", "enemy_behind", "skill_ready", "dash_ready"
]

def action_priority(a):
    if a.startswith("turn"):
        return 0
    if a.startswith("accelerate"):
        return 1
    if a.startswith("dash"):
        return 2
    return 3

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not all(col in data for col in feature_columns):
            return jsonify({"error": "Input JSON must contain all feature columns!"}), 400

        input_list = [float(data[col]) for col in feature_columns]
        input_tensor = torch.tensor([input_list], dtype=torch.float32)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).squeeze(0).numpy()

        # DEBUG
        print("INPUT:", input_list)
        print("PROBS:", np.round(probs, 3))
        print("idx2action:", idx2action)

        # ==== THRESHOLD LOWERED! ====
        threshold = 0.13
        chosen_idxs = [i for i, p in enumerate(probs) if p > threshold]

        if not chosen_idxs:
            chosen_idxs = [int(probs.argmax())]

        strategies = [idx2action[str(i)] for i in chosen_idxs]

        # ========== SORT action priority ==============
        strategies.sort(key=action_priority)
        print("STRATEGY:", strategies)
        # =============================================

        return jsonify({
            "strategy": strategies
        })
    except Exception as e:
        print("[API ERROR]", e)
        return jsonify({"strategy": ["stay"], "error": str(e)}), 500

if __name__ == '__main__':
    print("SLM API running at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
