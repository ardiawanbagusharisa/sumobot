# **Sumobot Seq2Seq SLM Pipeline**

---

## **Overview**

This repository contains a full-featured, modular pipeline for a **Small Language Model (SLM)** using a **Sequence-to-Sequence (Seq2Seq) architecture**.
It is designed to generate complex strategy strings for the Sumobot agent, given a natural language situation description or structured input.

**Key Features:**

* **Input:** Situation text (e.g. "Enemy is 1.2 meters ahead, angle 90 degrees, ...") or game feature vector (converted to text)
* **Output:** Auto-regressively generated multi-token strategy (e.g. "turn\_left\_90+dash+accelerate")
* **Model:** LSTM encoder-decoder (seq2seq) with token-based input/output
* **Tokenization:** Custom vocabulary for both input and output, supports special tokens (`<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`)
* **Deployment:** REST API for real-time inference

---

## **Main Scripts & Their Functions**

| Script                 | Description                                              |
| ---------------------- | -------------------------------------------------------- |
| `tokenizer_seq2seq.py` | Builds word-level tokenizers for both input and output.  |
| `seq2seq_model.py`     | Seq2Seq model (LSTM encoder-decoder, PyTorch).           |
| `train_seq2seq.py`     | Trains the seq2seq model on situation→strategy pairs.    |
| `generate_seq2seq.py`  | Generates strategies (token-by-token) from instructions. |
| `api_seq2seq.py`       | REST API for real-time inference (Flask-based).          |
| `tokenizer_utils.py`   | Utilities for vocabulary preview, encode/decode, OOVs.   |
| `vocab_seq2seq.json`   | Human-readable vocabulary dump (for reference only).     |

---

## **Workflow**

### **1. Tokenizer/Vocab Building**

```bash
python tokenizer_seq2seq.py
```

* Reads your merged dataset (`sumobot_slm_merged.csv`) with columns: `situation`, `strategy`.
* Tokenizes situation and strategy columns.
* Builds input/output vocabularies and saves as `tokenizer_seq2seq.pkl`.

---

### **2. Model Training**

```bash
python train_seq2seq.py
```

* Loads tokenized data and vocab.
* Trains the LSTM seq2seq model with teacher forcing.
* Outputs trained weights as `seq2seq_slm.pt`.

---

### **3. Generation/Inference**

```bash
python generate_seq2seq.py
```

* Loads model and tokenizer.
* Accepts a situation description (or feature vector formatted as text).
* Generates the strategy string **auto-regressively** (token by token, until `<EOS>`).
* Prints both generated token sequence and final strategy string.

---

### **4. REST API Serving**

```bash
python api_seq2seq.py
```

* Exposes a `/predict` endpoint.
* Accepts JSON POST with either all feature fields or a free-form situation text.
* Returns the generated strategy string (suitable for agent/game integration).

---

## **File List**

* `sumobot_slm_merged.csv`  – Merged dataset of situation/strategy pairs (input)
* `tokenizer_seq2seq.pkl`   – Saved tokenizer/vocab dictionary
* `vocab_seq2seq.json`      – (optional) Human-readable vocabulary
* `seq2seq_slm.pt`          – Trained model weights

---

## **Customization & Extensions**

* **Model:** You can swap in a Transformer encoder-decoder (or a deeper LSTM) in `seq2seq_model.py`.
* **Tokenizer:** For better OOV handling, consider using SentencePiece or HuggingFace tokenizers.
* **API:** Add endpoints or batch support for agent competitions.
* **Data:** Expand your dataset with more paraphrased instructions or richer context windows.

---