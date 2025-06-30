# **Sumobot SLM – Strategy Prediction Pipeline**

---

## **Overview**

This repository provides a complete, modular pipeline for building a **Small Language Model (SLM)**
that predicts multi-action strategies for Sumobot-style AI/robotics, based on tabular sensor input features.

* **Input:** Eight key numeric features per observation (distances, angles, booleans, etc.)
* **Output:** One or more strategy/action labels, e.g. `"turn_left_90+dash+accelerate"`
* **Model:** Shallow MLP classifier (can be extended)
* **Ready for:** Training, generation, and serving via API

---

## **Workflow**

### **1. Synthetic Dataset Generation**

**Script:** `generate_sumobot_dataset.py`

* Creates a CSV dataset with hybrid tabular + log-like features for Sumobot, including strategy combinations.
* Automatically generates context-like "situation" text for each sample.
* **Output:**

  * `sumobot_auto_dataset.csv` (primary dataset for training & testing)

---

### **2. Model Definition**

**Script:** `model.py`

* Defines the MLP (Multi-Layer Perceptron) classifier for mapping feature vectors to multi-label outputs.
* Simple, extensible, and compatible with PyTorch.

---

### **3. Model Training**

**Script:** `train.py`

* Loads and processes the dataset, splitting into training and validation sets.
* Converts string strategy labels into binary multi-label vectors.
* Trains the model using `BCEWithLogitsLoss` for multi-label output.
* Saves model weights and action label mappings for inference.
* **Output:**

  * `model.pt` (trained model)
  * `action2idx.json` / `idx2action.json` (mappings for API/generation)

---

### **4. Serving the Model (REST API)**

**Script:** `api.py`

* Loads the trained model and label mappings.
* Exposes a Flask-based API endpoint `/predict` for real-time inference.
* Accepts a POST request with all 8 numeric feature fields as JSON.
* Returns the selected strategies as a sorted list, based on action priority and probability thresholding.
* Ready for integration with Unity or other real-time systems.

---

## **Input/Output Example**

**API POST request:**

```json
{
  "enemy_distance": 0.9,
  "enemy_angle": 120,
  "edge_distance": 1.5,
  "center_distance": 3.2,
  "enemy_stuck": 0,
  "enemy_behind": 1,
  "skill_ready": 1,
  "dash_ready": 1
}
```

**API Response:**

```json
{
  "strategy": ["turn_left_90", "dash", "accelerate"]
}
```

---

## **File List**

| File                                  | Function                                           |
| ------------------------------------- | -------------------------------------------------- |
| `generate_sumobot_dataset.py`         | Generate dataset for training/testing              |
| `sumobot_auto_dataset.csv`            | Tabular + log-style dataset (generated)            |
| `model.py`                            | MLP classifier model (PyTorch)                     |
| `train.py`                            | Training script, outputs model + mappings          |
| `model.pt`                            | Saved PyTorch model weights                        |
| `action2idx.json` / `idx2action.json` | Strategy ↔ index mappings                          |
| `api.py`                              | Flask API for model inference (multi-label output) |

---

## **Quick Start**

1. **Generate dataset:**

   ```bash
   python generate_sumobot_dataset.py
   ```
2. **Train the model:**

   ```bash
   python train.py
   ```
3. **Run the API server:**

   ```bash
   python api.py
   ```
4. **Send a POST request to `/predict`** with all required feature fields.

---

## **Customizing / Extending**

* **Model:** Edit `model.py` for deeper/larger networks or feature engineering.
* **Thresholds:** Tune the probability threshold in `api.py` for more/less selective predictions.
* **Actions:** Dataset script and training pipeline handle any set of actions/combos.
* **Integration:** API designed for Unity, game, or robotics agent consumption.

---