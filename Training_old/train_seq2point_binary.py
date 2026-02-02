import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

CSV_PATH = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\train_mix_5s.csv"
OUT_PATH = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\models_trained_on_synth_sim_data"
# add near top
TARGET_COL = None  # ignore single target
WINDOW = 5                  # 5 bins = 25 seconds @ 5s
CENTER = WINDOW // 2
BATCH = 256
EPOCHS = 8
LR = 1e-3
SEED = 42

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

class Seq2PointDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)[:, None, :]  # (N, 1, T)
        self.y = torch.tensor(y, dtype=torch.float32)[:, None]     # (N, 1)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

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
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x

def make_windows(df, target_col, window):
    X_list, y_list = [], []
    for _, g in df.groupby("file"):
        p = g["P"].to_numpy(dtype=np.float32)
        y = g[target_col].to_numpy(dtype=np.float32)

        if len(p) < window:
            continue

        # sliding windows within the file
        for i in range(0, len(p) - window + 1):
            w = p[i:i+window].copy()
            # normalize per-window (helps a lot)
            w = (w - w.mean()) / (w.std() + 1e-6)
            X_list.append(w)
            y_list.append(y[i + window//2])

    X = np.stack(X_list) if X_list else np.zeros((0, window), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y

""" Train one model for one target column. the data is from synthetic dataset with multiple appliances states as targets. """
def train_one(df, target_col):
    df2 = df.dropna(subset=["P", target_col, "file"])
    X, y = make_windows(df2, target_col, WINDOW)
    if len(X) < 200 or y.sum() < 20:
        print(f"Skip {target_col}: too few samples/positives (N={len(X)}, pos={int(y.sum())})")
        return None

    idx = np.arange(len(X))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    train_loader = DataLoader(Seq2PointDataset(Xtr, ytr), batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(Seq2PointDataset(Xva, yva), batch_size=BATCH, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(WINDOW).to(device)
    os.makedirs(OUT_PATH, exist_ok=True)
    # imbalanced loss
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    pos_weight = torch.tensor([neg / (pos + 1e-6)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()

    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            prob = torch.sigmoid(model(xb.to(device))).cpu().numpy().reshape(-1)
            all_pred.append((prob >= 0.5).astype(np.int32))
            all_true.append(yb.numpy().reshape(-1).astype(np.int32))

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)

    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    # save model for this target
    os.makedirs(OUT_PATH, exist_ok=True)
    out = os.path.join(OUT_PATH, f"cnn_seq2point_{target_col}.pt")
    torch.save(model.state_dict(), out)
    print("Saved:", out)

    return {"target": target_col, "F1": f1, "P": prec, "R": rec, "pos_rate": float(y_true.mean())}

def main():
    df = pd.read_csv(CSV_PATH)
    y_cols = [c for c in df.columns if c.startswith("y_")]

    results = []
    for col in y_cols:
        r = train_one(df, col)
        if r is not None:
            results.append(r)

    res_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    res_path = os.path.join(OUT_PATH, "sim_synth_cnn_results.csv")
    res_df.to_csv(res_path, index=False)
    print("Wrote results:", res_path)
    print(res_df.head(10))
    
if __name__ == "__main__":
    main()
