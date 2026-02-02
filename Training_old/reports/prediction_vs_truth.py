import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

CSV_PATH = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\lit_synth_5s_states.csv"
MODEL_DIR = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports"

TARGET_COL = "y_N0"   # change: y_M0, y_H0, etc.
WINDOW = 5
THRESH = 0.5

OUT_PNG = os.path.join(MODEL_DIR, f"plot_pred_vs_truth_{TARGET_COL}.png")

class SmallCNN(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.net(x))

def make_windows_one_file(p, window):
    X = []
    centers = []
    for i in range(0, len(p) - window + 1):
        w = p[i:i+window].copy()
        w = (w - w.mean()) / (w.std() + 1e-6)
        X.append(w)
        centers.append(i + window//2)
    return np.stack(X), np.array(centers)

df = pd.read_csv(CSV_PATH)

# choose a file where this appliance is present at least once
cand_files = df.groupby("file")[TARGET_COL].sum()
cand_files = cand_files[cand_files > 0].index.tolist()
if not cand_files:
    raise RuntimeError(f"No files with any positives for {TARGET_COL}. Choose another target_col.")
file_id = cand_files[0]

g = df[df["file"] == file_id].sort_values("t_sec").reset_index(drop=True)

p = g["P"].to_numpy(np.float32)
y_true = g[TARGET_COL].to_numpy(np.float32)

X, centers = make_windows_one_file(p, WINDOW)
X_t = torch.tensor(X, dtype=torch.float32)[:, None, :]  # (N,1,T)

model = SmallCNN(WINDOW)
model_path = os.path.join(MODEL_DIR, f"cnn_seq2point_{TARGET_COL}.pt")
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

with torch.no_grad():
    probs = torch.sigmoid(model(X_t)).numpy().reshape(-1)

# build full-length prediction vector (NaN where we can't predict)
y_prob_full = np.full(len(p), np.nan, dtype=np.float32)
y_prob_full[centers] = probs
y_pred_full = (y_prob_full >= THRESH).astype(float)

plt.figure(figsize=(10,4))
plt.plot(g["t_sec"], g["P"])
plt.ylabel("Total Power (LF)")
plt.xlabel("Time (s)")
plt.title(f"{TARGET_COL}: Truth vs Model Prediction (one example file)")

# ground truth + prediction plotted at bottom
base = np.nanmin(g["P"])
rng = max(np.nanmax(g["P"]) - base, 1e-6)

plt.step(g["t_sec"], base + 0.10*rng*y_true, where="post", linewidth=2, label="Truth")
plt.step(g["t_sec"], base + 0.20*rng*y_pred_full, where="post", linewidth=2, label="Pred (0/1)")
plt.plot(g["t_sec"], base + 0.30*rng*y_prob_full, linewidth=2, label="Pred prob")

plt.legend()
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print("Saved:", OUT_PNG)
print("Example file:", file_id)
