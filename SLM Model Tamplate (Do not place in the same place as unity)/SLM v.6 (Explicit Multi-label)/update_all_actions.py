import pandas as pd
import numpy as np

# ==== CONFIG ====
DATA_CSV = "sumobot_context_dataset.csv"   
OLD_ACTIONS_TXT = "all_actions.txt"
NEW_ACTIONS_TXT = "all_actions_updated.txt"
OUTPUT_CSV = "sumobot_multilabel_dataset_updated.csv"
OUTPUT_NPY = "sumobot_multilabel_labels_updated.npy"

try:
    with open(OLD_ACTIONS_TXT, "r") as f:
        old_actions = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    old_actions = []
print(f"Loaded {len(old_actions)} old actions.")

df = pd.read_csv(DATA_CSV)
def extract_actions(strat_str):
    return [a.strip() for a in str(strat_str).split("+") if a.strip()]
data_actions = sorted(set(a for strat in df["strategy"] for a in extract_actions(strat)))
print(f"Found {len(data_actions)} actions in dataset.")

all_actions = sorted(set(old_actions) | set(data_actions))
print(f"Updated ALL_ACTIONS size: {len(all_actions)}")

with open(NEW_ACTIONS_TXT, "w") as f:
    for a in all_actions:
        f.write(f"{a}\n")
print(f"Saved updated ALL_ACTIONS to {NEW_ACTIONS_TXT}")

def strategy_to_multilabel_vector(strat_str, action_list):
    actions = set(extract_actions(strat_str))
    return [1 if a in actions else 0 for a in action_list]

ml_vectors = df["strategy"].apply(lambda x: strategy_to_multilabel_vector(x, all_actions))
ml_matrix = np.vstack(ml_vectors.values)
print("New multi-label matrix shape:", ml_matrix.shape)

out_df = df[["context_input"]].copy()
for i, action in enumerate(all_actions):
    out_df[action] = ml_matrix[:, i]

out_df.to_csv(OUTPUT_CSV, index=False)
np.save(OUTPUT_NPY, ml_matrix)
print(f"Saved updated multi-label dataset to {OUTPUT_CSV}")
print(f"Saved updated multi-label numpy array to {OUTPUT_NPY}")

print("\nSample output row after update:")
print(out_df.head(1))
