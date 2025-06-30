import pandas as pd

# 1. Load simulation data
df_sim = pd.read_csv("sumobot_auto_dataset.csv")

# 2. Convert numeric features to text instructions
def features_to_text(row):
    return (f"Enemy is {row['enemy_distance']} meters ahead, "
            f"angle {row['enemy_angle']} degrees, "
            f"edge distance {row['edge_distance']}, "
            f"center distance {row['center_distance']}, "
            f"enemy stuck: {bool(row['enemy_stuck'])}, "
            f"enemy behind: {bool(row['enemy_behind'])}, "
            f"skill ready: {bool(row['skill_ready'])}, "
            f"dash ready: {bool(row['dash_ready'])}")

df_sim['situation'] = df_sim.apply(features_to_text, axis=1)
df_sim_slm = df_sim[['situation', 'strategy']]
df_sim_slm['source'] = 'simulation'

# 3. Load battle log data
df_log = pd.read_csv("sumobot_slm_dataset.csv")
df_log['source'] = 'log'

# 4. Merge the datasets
df_total = pd.concat([df_sim_slm, df_log], ignore_index=True)
df_total = df_total.drop_duplicates() 

# 5. Save the combined dataset
df_total.to_csv("sumobot_slm_merged.csv", index=False)
print(f"Combined dataset is ready! Total rows: {len(df_total)}. File: sumobot_slm_merged.csv")
