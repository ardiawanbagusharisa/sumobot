import pandas as pd

# ========== CONFIG ==========
RAW_DATASET = "sumobot_slm_merged.csv" 
OUTPUT_CSV = "sumobot_context_dataset.csv"
N_CONTEXT = 3 
SEPARATOR = " [CTX] "

# ========== LOAD DATA ==========
df = pd.read_csv(RAW_DATASET)
if "situation" not in df.columns or "strategy" not in df.columns:
    raise Exception("Dataset must have columns 'situation' and 'strategy'!")

# ========== BUILD CONTEXT WINDOW ==========
contexts = []
strategies = []

history_buffer = []

for idx, row in df.iterrows():
    instruksi = str(row["situation"])
    history_buffer.append(instruksi)
    if len(history_buffer) > N_CONTEXT:
        history_buffer.pop(0)

    context_str = SEPARATOR.join(history_buffer)
    contexts.append(context_str)
    strategies.append(str(row["strategy"]))

# ========== SAVE NEW CONTEXTUAL DATASET ==========
df_out = pd.DataFrame({"context_input": contexts, "strategy": strategies})
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"Saved contextual dataset as {OUTPUT_CSV} with {len(df_out)} rows (context window: {N_CONTEXT})")
