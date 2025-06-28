import torch
import pickle
from seq2seq_model_context import Seq2SeqSLM_BiLSTM, Seq2SeqSLM_Transformer

# ==== CONFIG ====
MODEL_PATH = "seq2seq_context.pt"
TOKENIZER_PATH = "tokenizer_seq2seq_context.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN_IN = 32
MAX_LEN_OUT = 16
USE_TRANSFORMER = False    
SAMPLING_K = 3                
SAMPLING_TEMP = 1.0        
PRINT_STEP = True          

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

# ==== GENERATE STRATEGY ====
def generate_strategy(context_str, max_gen_len=MAX_LEN_OUT, do_sampling=True, print_step=PRINT_STEP):
    tokens_in = tokenize_context(context_str)
    ids_in = encode_tokens(tokens_in, token2idx_in, "<UNK>", MAX_LEN_IN)
    src = torch.tensor([ids_in], dtype=torch.long).to(DEVICE)
    # Inference: always auto-regressive
    with torch.no_grad():
        outputs = model(src, trg=None, do_sampling=do_sampling, max_len=max_gen_len, start_token_idx=BOS_IDX)
        logits = outputs[0]  # (max_len, vocab_out)
       
        pred_tokens = []
        for i, logit in enumerate(logits):
            if do_sampling and SAMPLING_K > 1:
               
                probs = torch.softmax(logit / SAMPLING_TEMP, dim=-1)
                topk_probs, topk_idx = torch.topk(probs, SAMPLING_K)
                next_token_idx = topk_idx[torch.multinomial(topk_probs, 1)].item()
            else:
                next_token_idx = logit.argmax(-1).item()
            next_token = idx2token_out.get(next_token_idx, "<UNK>")
            if print_step:
                print(f"Step {i+1}: {next_token}")
            if next_token == "<EOS>":
                break
            pred_tokens.append(next_token)

        if pred_tokens and pred_tokens[0] == "<BOS>":
            pred_tokens = pred_tokens[1:]

    strategy = ""
    for tok in pred_tokens:
        if tok == "+":
            strategy += "+"
        else:
            if strategy and not strategy.endswith("+"):
                strategy += "+"
            strategy += tok
    strategy = strategy.replace("++", "+").strip("+")
    return strategy, pred_tokens

# ==== EXAMPLE USAGE ====
if __name__ == "__main__":
    context = "Enemy is 1.1 meters ahead angle 90 degrees dash ready [CTX] Enemy is 1.0 meters ahead angle 85 degrees dash ready [CTX] Enemy is 0.8 meters ahead angle 80 degrees dash ready"
    print("\n===== Generating strategy with top-k sampling =====")
    strategy, tokens = generate_strategy(context, do_sampling=True)
    print("Context input:", context)
    print("Generated tokens:", tokens)
    print("Generated strategy string:", strategy)

    print("\n===== Generating strategy with greedy/argmax =====")
    strategy2, tokens2 = generate_strategy(context, do_sampling=False)
    print("Generated tokens:", tokens2)
    print("Generated strategy string:", strategy2)
