import pandas as pd
import numpy as np

# ========= CONFIG =========
INPUT_CSV = "sumobot_context_dataset.csv" 
OUTPUT_CSV = "sumobot_multilabel_dataset.csv"
OUTPUT_NPY = "sumobot_multilabel_labels.npy"
ACTION_LIST_TXT = "all_actions.txt"

df = pd.read_csv(INPUT_CSV)

def extract_actions(strat_str):
    return [a.strip() for a in strat_str.split("+") if a.strip()]

all_actions = sorted(set(a for strat in df["strategy"] for a in extract_actions(strat)))
print(f"Total unique actions: {len(all_actions)}")
print("ALL_ACTIONS:", all_actions)

with open(ACTION_LIST_TXT, "w") as f:
    for a in all_actions:
        f.write(f"{a}\n")

def strategy_to_multilabel_vector(strat_str, action_list):
    actions = set(extract_actions(strat_str))
    return [1 if a in actions else 0 for a in action_list]

ml_vectors = df["strategy"].apply(lambda x: strategy_to_multilabel_vector(x, all_actions))
ml_matrix = np.vstack(ml_vectors.values)

print("Multi-label matrix shape:", ml_matrix.shape)

out_df = df[["context_input"]].copy()
for i, action in enumerate(all_actions):
    out_df[action] = ml_matrix[:, i]

out_df.to_csv(OUTPUT_CSV, index=False)
np.save(OUTPUT_NPY, ml_matrix)
print(f"Saved multi-label dataset to {OUTPUT_CSV}")
print(f"Saved multi-label numpy array to {OUTPUT_NPY}")
print(f"Saved ALL_ACTIONS to {ACTION_LIST_TXT}")

print("\nSample output row:")
print(out_df.head(1))
