import torch
import pickle
import pandas as pd
from seq2seq_model_context import Seq2SeqSLM_BiLSTM, Seq2SeqSLM_Transformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ==== CONFIG ====
MODEL_PATH = "seq2seq_context.pt"
TOKENIZER_PATH = "tokenizer_seq2seq_context.pkl"
DATA_PATH = "sumobot_context_dataset.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN_IN = 32
MAX_LEN_OUT = 16
USE_TRANSFORMER = False
SAMPLING_K = 1  
SAMPLING_TEMP = 1.0
EVAL_SAMPLES = 200  

# ==== LOAD TOKENIZER & MODEL ====
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

if USE_TRANSFORMER:
    model = Seq2SeqSLM_Transformer(
        vocab_in_size, vocab_out_size, emb_dim=128, nhead=4, num_layers=2,
        pad_idx_in=PAD_IDX_IN, pad_idx_out=token2idx_out["<PAD>"],
        sampling_k=SAMPLING_K, sampling_temp=SAMPLING_TEMP
    ).to(DEVICE)
else:
    model = Seq2SeqSLM_BiLSTM(
        vocab_in_size, vocab_out_size, emb_dim=128, hidden_dim=256,
        pad_idx_in=PAD_IDX_IN, pad_idx_out=token2idx_out["<PAD>"],
        sampling_k=SAMPLING_K, sampling_temp=SAMPLING_TEMP
    ).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==== TOKENIZER UTILS ====
def tokenize_context(context_str):
    return context_str.lower().split()

def encode_tokens(seq, token2idx, unk_token="<UNK>", maxlen=None):
    ids = [token2idx.get(tok, token2idx.get(unk_token, 1)) for tok in seq]
    if maxlen:
        pad_idx = token2idx.get("<PAD>", 0)
        ids = ids[:maxlen] + [pad_idx]*(maxlen - len(ids)) if len(ids) < maxlen else ids[:maxlen]
    return ids

def decode_ids(ids, idx2token):
    return [idx2token.get(i, "<UNK>") for i in ids]

def tokenize_strategy(strategy_str):
    tokens = []
    for tok in strategy_str.replace('+', ' + ').split():
        tokens.append(tok.strip())
    return ["<BOS>"] + tokens + ["<EOS>"]

# ==== EVALUASI TOKEN-BY-TOKEN & BLEU ====
def eval_seq2seq(samples=EVAL_SAMPLES):
    df = pd.read_csv(DATA_PATH)
    if samples < len(df):
        df = df.sample(n=samples, random_state=42)
    total_tokens = 0
    correct_tokens = 0
    bleu_scores = []

    for idx, row in df.iterrows():
        # --- Build input ---
        context_str = str(row["context_input"])
        true_strategy = str(row["strategy"])
        # --- Encode input ---
        tokens_in = tokenize_context(context_str)
        ids_in = encode_tokens(tokens_in, token2idx_in, "<UNK>", MAX_LEN_IN)
        src = torch.tensor([ids_in], dtype=torch.long).to(DEVICE)
        # --- Target tokens ---
        true_tokens = tokenize_strategy(true_strategy)
        true_tokens = true_tokens[1:-1]  # remove <BOS> and <EOS> for eval
        # --- Model inference ---
        with torch.no_grad():
            outputs = model(src, trg=None, do_sampling=(SAMPLING_K > 1), max_len=MAX_LEN_OUT, start_token_idx=BOS_IDX)
            logits = outputs[0]  # (max_len, vocab_out)
            # Greedy argmax
            pred_ids = logits.argmax(-1).cpu().tolist()
            pred_tokens = []
            for tok_id in pred_ids:
                tok = idx2token_out.get(tok_id, "<UNK>")
                if tok == "<EOS>":
                    break
                pred_tokens.append(tok)
            # Remove <BOS> jika ada
            if pred_tokens and pred_tokens[0] == "<BOS>":
                pred_tokens = pred_tokens[1:]
        # --- Token-level accuracy ---
        min_len = min(len(pred_tokens), len(true_tokens))
        match = sum([1 if i < len(pred_tokens) and pred_tokens[i] == true_tokens[i] else 0 for i in range(min_len)])
        total_tokens += len(true_tokens)
        correct_tokens += match
        # --- BLEU Score ---
        cc = SmoothingFunction()
        bleu = sentence_bleu([true_tokens], pred_tokens, weights=(0.5, 0.5), smoothing_function=cc.method1)
        bleu_scores.append(bleu)
        # --- Print sample ---
        if idx < 5:  # Print 5 sample
            print(f"\n[{idx}]")
            print("Context :", context_str)
            print("Target  :", true_tokens)
            print("Predict :", pred_tokens)
            print(f"Token match: {match}/{len(true_tokens)} | BLEU: {bleu:.3f}")

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Token accuracy: {100*correct_tokens/total_tokens:.2f}% ({correct_tokens}/{total_tokens})")
    print(f"BLEU score (mean): {sum(bleu_scores)/len(bleu_scores):.3f}")
    print("Sample:", len(df))

if __name__ == "__main__":
    eval_seq2seq()
