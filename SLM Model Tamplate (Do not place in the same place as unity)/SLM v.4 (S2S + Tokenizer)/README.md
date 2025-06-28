# **Sumobot Context-Aware Seq2Seq SLM Pipeline**

---

## **Overview**

This repository implements a complete, context-aware **Sequence-to-Sequence (Seq2Seq) Small Language Model (SLM)** for generating strategies in the Sumobot environment.
The model is trained to output token-based strategy sequences based on a *context window*—multiple recent situation texts, not just the current one.

---

## **Features**

* **Contextual Input:** Uses a sliding window of N previous situations as input, separated by `[CTX]`.
* **Token-Based Seq2Seq Model:** LSTM (or Transformer) encoder-decoder with custom vocabulary.
* **Auto-regressive Generation:** Generates strategy tokens step by step.
* **Plug-and-Play API:** REST endpoint for real-time strategy generation, supports both string and list inputs.
* **Unity Integration:** Includes a Python class for managing context history buffers.

---

## **Pipeline & Scripts**

### **1. Prepare Contextual Dataset**

**Script:** `prepare_contextual_dataset.py`

* Builds context windows from a merged dataset of situation-strategy pairs (e.g., `sumobot_slm_merged.csv`).
* **Output:** `sumobot_context_dataset.csv` with columns `context_input` and `strategy`.

### **2. Build and Save Tokenizer**

**Script:** `tokenizer_seq2seq_context.py`

* Tokenizes context inputs and strategy outputs, building input/output vocabularies.
* Saves as `tokenizer_seq2seq_context.pkl` and `vocab_seq2seq_context.json`.

### **3. Train Seq2Seq Model**

**Script:** `train_seq2seq_context.py`

* Loads the contextual dataset and tokenizer.
* Trains a BiLSTM or Transformer encoder-decoder (switch with a flag).
* Uses teacher forcing for faster convergence.
* **Output:** `seq2seq_context.pt` (PyTorch model weights).

### **4. Generate / Inference**

**Script:** `generate_seq2seq_context.py`

* Loads model and tokenizer.
* Accepts a full context string as input.
* Generates a token sequence as strategy (auto-regressively, until `<EOS>`).
* Prints both the sequence of generated tokens and the final strategy string.

### **5. Serve with REST API**

**Script:** `api_seq2seq_context.py`

* Runs a Flask server with `/predict` endpoint.
* Accepts either:

  * `context_input` as a string (context window separated by `[CTX]`), **or**
  * `context_input` as a list (history buffer)
* Returns a JSON with the generated strategy string.

**Example API input:**

```json
{ "context_input": "Enemy is 1.1 meters ahead ... [CTX] Enemy is 1.0 meters ahead ... [CTX] ..." }
```

**or:**

```json
{ "context_input": ["Enemy is 1.1 meters ahead ...", "Enemy is 1.0 meters ahead ...", ...] }
```

**API output:**

```json
{ "strategy": "turn_left_90+dash+accelerate" }
```

### **6. Unity/Agent Context Buffer**

**Script:** `context_buffer.py`

* `ContextBuffer` class for managing a fixed-length sliding window of situation strings (history) on the agent side.
* Produces either a string (with `[CTX]` separator) or a list for direct API calls.

---

## **How to Run**

1. **Prepare your merged dataset:**
   Ensure you have `sumobot_slm_merged.csv` (columns: `situation`, `strategy`).

2. **Build context windows:**

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

5. **Test generation:**

   ```bash
   python generate_seq2seq_context.py
   ```

6. **Run API server:**

   ```bash
   python api_seq2seq_context.py
   ```

---

## **File List**

* `prepare_contextual_dataset.py` — Build context windows for training
* `sumobot_context_dataset.csv` — Final dataset with context windows
* `tokenizer_seq2seq_context.py` — Tokenizer and vocabulary builder
* `vocab_seq2seq_context.json` / `tokenizer_seq2seq_context.pkl` — Vocab/tokenizer files
* `seq2seq_model_context.py` — Model definitions (BiLSTM/Transformer encoder-decoder)
* `train_seq2seq_context.py` — Training script
* `seq2seq_context.pt` — Trained model weights
* `generate_seq2seq_context.py` — Manual/test inference script
* `api_seq2seq_context.py` — REST API for deployment
* `context_buffer.py` — Unity/agent-side context history manager

---

## **Customizing**

* **Context Length:** Set `N_CONTEXT` in `prepare_contextual_dataset.py` and in your agent buffer.
* **Model:** Use BiLSTM (default) or Transformer by changing `USE_TRANSFORMER` flag.
* **Tokenization:** The pipeline supports word-level by default, can be extended to BPE/subword easily.
* **Strategy Format:** Model learns and generates composite strategies, e.g. `"turn_left_90+dash+accelerate"`.

---