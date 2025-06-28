
# **SLM Multi-label Context-Aware Pipeline**


---

## **Overview**

This repository contains a modular **Small Language Model (SLM) multi-label context-aware pipeline**
for strategy/action prediction in games or robotics, based on observation history in natural language.

**Key Features:**

* **Input:** Context window (multiple-step history), natural language format, `[CTX]` as a separator
* **Output:** One or more actions (multi-label), as a string (e.g. `turn_left_90+dash`)
* **Model:** Encoder (BiLSTM or Transformer) + sigmoid output for multi-label classification
* **Evaluation:** F1-score, precision, recall, per-action classification report
* **Deployment:** REST API for integration with Unity or other agents

---

## **Workflow**

### **1. Prepare the Initial Dataset**

A simple CSV file with at least two columns:

| context\_input                   | strategy                   |
| -------------------------------- | -------------------------- |
| Enemy is ... \[CTX] Enemy is ... | turn\_left\_90+dash        |
| Enemy is ... \[CTX] Enemy is ... | turn\_right\_45+accelerate |

---

### **2. Build the Multi-label Dataset**

```bash
python prepare_multilabel_dataset.py
```

**Output:**

* `sumobot_multilabel_dataset.csv`
* `sumobot_multilabel_labels.npy`
* `all_actions.txt`

---

### **3. Tokenize Context Input**

```bash
python tokenizer_multilabel.py
```

**Output:**

* `tokenizer_multilabel.pkl`
* `vocab_multilabel.json`

---

### **4. Train the Multi-label Model**

```bash
python train_multilabel.py
```

**Output:**

* `slm_multilabel.pt` (trained model, ready for deployment and inference)

---

### **5. Generate/Preview Model Output**

```bash
python generate_multilabel.py
```

* Input any `context_input`
* Output: predicted strategy string (multi-label) and per-action confidence scores

---

### **6. Evaluate the Model**

```bash
python eval_multilabel.py
```

* Outputs:

  * F1-score, precision, recall, per-action report
  * Example predictions vs. targets

---

### **7. Deploy as REST API**

```bash
python api_multilabel.py
```

**Endpoint:** `/predict` (POST)

* **Input:**

  ```json
  {
    "context_input": "Enemy is ... [CTX] Enemy is ...",
    "threshold": 0.5
  }
  ```
* **Output:**

  ```json
  { "strategy": "turn_left_90+dash" }
  ```

---

## **File Structure**

```
prepare_multilabel_dataset.py
update_all_actions.py
tokenizer_multilabel.py
slm_multilabel_model.py
train_multilabel.py
generate_multilabel.py
eval_multilabel.py
api_multilabel.py
sumobot_multilabel_dataset.csv
sumobot_multilabel_labels.npy
all_actions.txt
tokenizer_multilabel.pkl
slm_multilabel.pt
```

---

## **Tips & Notes**

* If you add more data or new actions, run `update_all_actions.py` and retrain the model
* The default action prediction threshold is 0.5 (can be changed in the scripts or API)
* All scripts are modular—run each step independently as needed
* The API is ready for real-time integration with Unity/agents

---

## **Roadmap for Future Upgrades**

* Upgrade to BPE/subword tokenizer (e.g., SentencePiece)
* Add Transformer or contextual encoder (e.g., BERT)
* Multi-tasking (add task type label in context\_input)
* Continuous learning (automated retraining from new log data)

---