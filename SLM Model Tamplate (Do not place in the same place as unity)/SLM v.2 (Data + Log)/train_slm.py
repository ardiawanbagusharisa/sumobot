import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from slm_model import SimpleTokenizer, SLMModel

# ====== 1. Load Dataset ======
df = pd.read_csv("sumobot_slm_merged.csv")
texts = df["situation"].tolist()
labels = df["strategy"].tolist()

# Mapping aksi ke index
all_actions = sorted(set(labels))
action2idx = {a: i for i, a in enumerate(all_actions)}
idx2action = {i: a for a, i in action2idx.items()}
y = [action2idx[a] for a in labels]

# ====== 2. Tokenizer & Vocab ======
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(texts)
vocab_size = len(tokenizer.word2idx)
max_len = max(len(t.split()) for t in texts)  # bisa dibatasi, ex: 20

# Encode input
counts = Counter(labels)
min_count = 2
filtered_data = [(t, a) for t, a in zip(texts, labels) if counts[a] >= min_count]
if not filtered_data:
    raise Exception("Dataset is too small after rare class filter! Add more data.")
texts, labels = zip(*filtered_data)
y = [action2idx[a] for a in labels]
X = [tokenizer.encode(t, max_len) for t in texts]
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# ====== 3. Train/Test Split ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# ====== 4. Model, Loss, Optim ======
emb_dim = 64
hidden_dim = 128
output_dim = len(all_actions)
model = SLMModel(vocab_size, emb_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ====== 5. Training Loop ======
EPOCHS = 30
batch_size = 16

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    train_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x = X_train[indices]
        batch_y = y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        val_loss = criterion(outputs, y_test)
        preds = outputs.argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {acc:.4f}")

# ====== 6. Save Model & Mapping ======
torch.save(model.state_dict(), "slm_model.pt")
import json
with open("action2idx.json", "w") as f:
    json.dump(action2idx, f)
with open("idx2action.json", "w") as f:
    json.dump(idx2action, f)
import pickle
with open("slm_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("SLM model, action mapping, and tokenizer saved!")

