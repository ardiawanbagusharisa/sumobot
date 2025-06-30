import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from collections import Counter
from sklearn.model_selection import train_test_split

from model import StrategyEmbeddingModel

# ===== 1. Load Data & Preprocess =====

df = pd.read_csv("sumobot_auto_dataset.csv") 

feature_columns = [
    "enemy_distance", "enemy_angle", "edge_distance", "center_distance",
    "enemy_stuck", "enemy_behind", "skill_ready", "dash_ready"
]

X = df[feature_columns].astype(float).values.tolist()

all_actions = set()
for strat in df["strategy"]:
    for act in strat.split("+"):
        all_actions.add(act.strip())
all_actions = sorted(list(all_actions)) 

action2idx = {a: i for i, a in enumerate(all_actions)}
idx2action = {i: a for i, a in enumerate(all_actions)}

Y = []
for strat in df["strategy"]:
    acts = strat.split("+")
    label_vec = [0] * len(all_actions)
    for act in acts:
        idx = action2idx[act.strip()]
        label_vec[idx] = 1
    Y.append(label_vec)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

# ===== 2. Train/Test Split =====

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42
)

# ===== 3. Model, Loss, Optimizer =====

input_dim = X.shape[1]
output_dim = Y.shape[1]
hidden_dim = 32 
model = StrategyEmbeddingModel(input_dim, hidden_dim, output_dim)

criterion = nn.BCEWithLogitsLoss()  # multilabel
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===== 4. Training Loop =====

EPOCHS = 150
batch_size = 8

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    train_loss = 0.0

    for i in range(0, X_train.size(0), batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], Y_train[indices]

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation loss
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        val_loss = criterion(outputs, Y_test)
    
    if epoch % 10 == 0 or epoch == EPOCHS-1:
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

# ===== 5. Save Model & Mapping =====

torch.save(model.state_dict(), "model.pt")
with open("action2idx.json", "w") as f:
    json.dump(action2idx, f)
with open("idx2action.json", "w") as f:
    json.dump(idx2action, f)

print("Training is complete. Model and action mapping have been saved..")

print("Distribution of action:")
print(Counter(df["strategy"]))
