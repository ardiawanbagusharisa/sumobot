# **Sumobot SLM – Strategy Language Model Pipeline With Data Log**

---

## **Overview**

This repository implements a classic **Small Language Model (SLM)** pipeline for predicting strategies in a Sumobot environment.
The model receives either raw numeric features or a situation “template” text as input and predicts the best strategy/action combination from a defined set.

**Key Features:**

* **Input:** Numeric feature vector OR text template describing the situation
* **Output:** One best-fit strategy/action (multi-class classification)
* **Model:** LSTM encoder (PyTorch)
* **Serving:** REST API with JSON interface for integration with games, robotics, or simulation

---

## **Workflow**

### **1. Dataset Preparation**

* Prepare a CSV file (e.g., `sumobot_slm_merged.csv`) with at least:

  * `situation` (template-based textual description of the current environment)
  * `strategy` (single action/strategy label from your defined set)

---

### **2. Training the Model**

**Script:** `train_slm.py`

* Loads and preprocesses the dataset.
* Tokenizes situation texts and builds vocabulary (`SimpleTokenizer`).
* Converts strategies to integer class labels (`action2idx.json`, `idx2action.json`).
* Trains an LSTM encoder + linear classifier (`SLMModel`).
* Evaluates and prints loss & accuracy per epoch.
* **Outputs:**

  * `slm_model.pt` (PyTorch weights)
  * `slm_tokenizer.pkl` (tokenizer/vocab)
  * `action2idx.json`, `idx2action.json` (label mappings)

---

### **3. REST API for Inference**

**Script:** `api_slm.py`

* Loads the trained model, tokenizer, and action mappings.
* Exposes a `/predict` endpoint:

  * **Input:** JSON with either all required numeric features, or a situation string.
  * **Processing:** Converts features to a template string if needed, tokenizes, encodes, and infers the best strategy.
  * **Output:** JSON with the predicted strategy string.
* Can be integrated with Unity, robot agents, or any client able to send HTTP POST requests.

---

## **File Overview**

| File                | Purpose                                                 |
| ------------------- | ------------------------------------------------------- |
| `slm_model.py`      | Model + tokenizer definitions                           |
| `train_slm.py`      | Training pipeline, tokenizer build, save model/mappings |
| `api_slm.py`        | Flask REST API for inference                            |
| `action2idx.json`   | Mapping: strategy string → index                        |
| `idx2action.json`   | Mapping: index → strategy string                        |
| `slm_tokenizer.pkl` | Saved tokenizer/vocab object                            |
| `slm_model.pt`      | Saved PyTorch model weights                             |

---

## **Customization**

* **Model:**
  Edit `slm_model.py` for a deeper LSTM, different embedding size, or custom architecture.
* **Tokenizer:**
  Simple word-level by default. For better generalization, switch to BPE/SentencePiece.
* **Action Set:**
  Fully extensible; just update your data and retrain.

---
