# **Sumobot Context-Aware Seq2Seq SLM Pipeline**

---

## **Overview**

This repository implements a **Context-Aware Sequence-to-Sequence Small Language Model (SLM)** for the Sumobot environment.
It is designed to generate complex, auto-regressive strategy sequences based on a context window of recent situation descriptions—enabling more adaptive and history-aware decision-making for agents or robots.

---

## **Features**

* **Context Window Input:** The model considers multiple past situations (context), not just the latest step.
* **Auto-Regressive Seq2Seq Model:** BiLSTM or Transformer encoder-decoder with custom tokenization.
* **Flexible Inference:** Top-k sampling or greedy decoding.
* **Plug-and-Play REST API:** For integration with Unity, games, or other agent controllers.
* **Evaluation & Visualization:** Includes scripts for evaluation, generation preview, and context buffer management.

---

## **Workflow**

### **1. Prepare Contextual Dataset**

**Script:** `prepare_contextual_dataset.py`

* Converts a dataset of situation-strategy pairs (e.g., `sumobot_slm_merged.csv`) into context windows, where each row includes multiple past situations (separated by `[CTX]`).
* **Output:** `sumobot_context_dataset.csv`.

---

### **2. Tokenizer and Vocabulary**

**Script:** `tokenizer_seq2seq_context.py`

* Tokenizes both context inputs and strategy outputs.
* Builds word-level vocabularies, including special tokens (`<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`, `[CTX]`).
* **Output:** `tokenizer_seq2seq_context.pkl` (main vocab), `vocab_seq2seq_context.json` (for reference).

---

### **3. Train the Seq2Seq Model**

**Script:** `train_seq2seq_context.py`

* Loads the contextual dataset and tokenizer.
* Trains a BiLSTM or Transformer encoder-decoder model using teacher-forcing and auto-regressive outputs.
* Supports top-k sampling during training for evaluation.
* **Output:** `seq2seq_context.pt` (PyTorch model).

---

### **4. Generation and Inference**

**Script:** `generate_seq2seq_context.py`

* Loads the trained model and tokenizer.
* Accepts a context window as input and generates the corresponding strategy sequence (token by token).
* Supports both greedy and top-k sampling generation.

---

### **5. REST API for Real-time Inference**

**Script:** `api_seq2seq_context.py`

* Exposes a Flask API endpoint `/predict` for online inference.
* Input:

  * `context_input` as a string (context window with `[CTX]` separator), **or**
  * `context_input` as a list (history buffer).
* Output:

  * JSON with the generated strategy string.

**Example request:**

```json
{ "context_input": "Enemy is 1.1 meters ahead ... [CTX] Enemy is 1.0 meters ahead ..." }
```

**or:**

```json
{ "context_input": ["Enemy is 1.1 meters ahead ...", "Enemy is 1.0 meters ahead ..."] }
```

**Response:**

```json
{ "strategy": "turn_left_90+dash+accelerate" }
```

---

### **6. Evaluation Script**

**Script:** `eval_seq2seq.py`

* Evaluates the trained model using token-level accuracy and BLEU scores.
* Prints example predictions vs. ground truth for inspection.

---

### **7. Unity/Agent Integration – Context Buffer**

**Script:** `context_buffer.py`

* Contains a `ContextBuffer` class to manage the sliding window of past situations (for use in Unity, Python agent, or testing).
* Produces either a concatenated string or a list for easy API consumption.

---

## **How to Use**

1. **Prepare your merged situation-strategy dataset** as `sumobot_slm_merged.csv`.
2. **Create context windows:**

   ```bash
   python prepare_contextual_dataset.py
   ```
3. **Build tokenizer:**

   ```bash
   python tokenizer_seq2seq_context.py
   ```
4. **Train the model:**

   ```bash
   python train_seq2seq_context.py
   ```
5. **Preview generation (optional):**

   ```bash
   python generate_seq2seq_context.py
   ```
6. **Evaluate the model (optional):**

   ```bash
   python eval_seq2seq.py
   ```
7. **Run API server:**

   ```bash
   python api_seq2seq_context.py
   ```
8. **(Unity/agent) Use `context_buffer.py` to manage context history before sending to API.**

---

## **Files**

* `prepare_contextual_dataset.py`
* `sumobot_context_dataset.csv`
* `tokenizer_seq2seq_context.py`, `tokenizer_seq2seq_context.pkl`, `vocab_seq2seq_context.json`
* `seq2seq_model_context.py`
* `train_seq2seq_context.py`, `seq2seq_context.pt`
* `generate_seq2seq_context.py`
* `eval_seq2seq.py`
* `api_seq2seq_context.py`
* `context_buffer.py`

---

## **Customizing**

* Change the context window length (`N_CONTEXT`) in `prepare_contextual_dataset.py` and in your agent.
* Switch between BiLSTM and Transformer models with the `USE_TRANSFORMER` flag.
* Adjust sampling temperature and top-k in scripts for more/less diverse output.

---