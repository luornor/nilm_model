# finetune_natural_selftrain_fixed.py
import os, re
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# === PATHS ===
NAT_PREDS = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\natural_preds.csv"
# MODEL_IN  = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\cnn_seq2point_y_AC_Adapter_Sony_M0.pt"
# MODEL_IN = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\cnn_seq2point_y_Soldering_Station_H0.pt"
MODEL_IN = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\cnn_seq2point_y_Incandescent_Lamp_N0.pt"
MODEL_OUT = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\cnn_seq2point_y_Incandescent_Lamp_N0_finetuned_natural.pt"

# === SETTINGS ===
WINDOW = 5
BATCH  = 256
EPOCHS = 5
LR     = 2e-4
SEED   = 42

# self-training thresholds (tune if needed)
POS_THR = 0.90
NEG_THR = 0.10
MAX_POS = 6000
MAX_NEG = 6000
MIN_CONFIDENT = 200  # require at least this many positives and negatives (fallback if not reached)

torch.manual_seed(SEED)
np.random.seed(SEED)

class Seq2PointDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)[:, None, :]
        self.y = torch.tensor(y, dtype=torch.float32)[:, None]

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 1))

    def forward(self, x): return self.head(self.net(x))

def find_prob_col(df: pd.DataFrame, model_path: str) -> str:
    base = os.path.basename(model_path)
    m = re.search(r"(y_[A-Za-z0-9_]+)", base)
    token = m.group(1) if m else None
    cols = df.columns.tolist()
    candidates = []
    if token:
        token2 = token.replace("y_", "")
        for c in cols:
            cl = c.lower()
            if ("prob" in cl or "pred" in cl or "p_" in cl) and (token.lower() in cl or token2.lower() in cl):
                candidates.append(c)
        if not candidates:
            for c in cols:
                cl = c.lower()
                if token.lower() in cl or token2.lower() in cl:
                    candidates.append(c)
    if not candidates:
        for c in cols:
            cl = c.lower()
            if "prob" in cl and cl not in {"prob"}:
                candidates.append(c)
    if not candidates:
        raise ValueError(f"Couldn't find probability column. Columns: {cols[:30]} ...")
    return candidates[0]

def make_windows(p: np.ndarray, window: int):
    X = []
    for i in range(0, len(p) - window + 1):
        w = p[i:i+window].astype(np.float32).copy()
        w = (w - w.mean()) / (w.std() + 1e-6)
        X.append(w)
    return np.stack(X) if X else np.zeros((0, window), np.float32)

def main():
    df = pd.read_csv(NAT_PREDS)
    if "P" not in df.columns:
        raise ValueError("natural_preds.csv must include column 'P'.")
    prob_col = find_prob_col(df, MODEL_IN)
    print("Using probability column:", prob_col)

    # collect windows and aligned probs across all files
    X_all = []
    prob_centers_all = []

    if "file" in df.columns:
        groups = df.groupby("file")
    else:
        groups = [("all", df)]

    for fname, g in groups:
        g = g.sort_values("t_sec") if "t_sec" in g.columns else g
        p = g["P"].to_numpy(dtype=np.float32)
        prob = g[prob_col].to_numpy(dtype=np.float32) if prob_col in g.columns else np.full(len(p), np.nan, dtype=np.float32)

        if len(p) < WINDOW:
            continue
        X = make_windows(p, WINDOW)           # shape (Nw, W)
        center = WINDOW // 2
        # align: centers correspond to indices center .. center+Nw-1 in original p
        prob_center = prob[center: center + len(X)]
        # keep only windows that have non-NaN prob_center
        mask = ~np.isnan(prob_center)
        if mask.sum() == 0:
            continue
        X_all.append(X[mask])
        prob_centers_all.append(prob_center[mask])

    if not X_all:
        raise RuntimeError("No windows produced across all files. Check natural_preds.csv and prob column.")

    X_all = np.concatenate(X_all, axis=0)
    prob_centers = np.concatenate(prob_centers_all, axis=0)

    print("Total windows:", len(X_all))
    print("Prob stats: min=%.3f mean=%.3f max=%.3f std=%.3f (NaNs removed)" %
          (float(np.nanmin(prob_centers)), float(np.nanmean(prob_centers)), float(np.nanmax(prob_centers)), float(np.nanstd(prob_centers))))

    # high-confidence selection by thresholds
    pos_idx = np.where(prob_centers >= POS_THR)[0]
    neg_idx = np.where(prob_centers <= NEG_THR)[0]
    print(f"Initial selected: pos={len(pos_idx)}, neg={len(neg_idx)} using POS_THR={POS_THR}, NEG_THR={NEG_THR}")

    # fallback: if thresholds too strict, pick top-K and bottom-K by probability
    if len(pos_idx) < MIN_CONFIDENT or len(neg_idx) < MIN_CONFIDENT:
        print("Threshold selection insufficient. Using top/bottom fallback selection.")
        # choose up to MAX_POS/NEG or MIN_CONFIDENT whichever is larger but within available
        kpos = min(MAX_POS, max(len(prob_centers)//10, MIN_CONFIDENT))  # reasonable fallback sizes
        kneg = min(MAX_NEG, max(len(prob_centers)//10, MIN_CONFIDENT))
        sorted_idx = np.argsort(prob_centers)
        neg_idx_fb = sorted_idx[:kneg]
        pos_idx_fb = sorted_idx[-kpos:]
        # ensure we actually have different indices
        pos_idx = np.unique(np.concatenate([pos_idx, pos_idx_fb]))
        neg_idx = np.unique(np.concatenate([neg_idx, neg_idx_fb]))
        print(f"Fallback selected: pos={len(pos_idx)}, neg={len(neg_idx)}")

    # cap
    rng = np.random.default_rng(SEED)
    if len(pos_idx) > MAX_POS:
        pos_idx = rng.choice(pos_idx, size=MAX_POS, replace=False)
    if len(neg_idx) > MAX_NEG:
        neg_idx = rng.choice(neg_idx, size=MAX_NEG, replace=False)

    if len(pos_idx) < 50 or len(neg_idx) < 50:
        raise RuntimeError(f"Not enough confident samples after fallback. pos={len(pos_idx)}, neg={len(neg_idx)}. Try lowering POS_THR/raising NEG_THR or ensure natural_preds has valid probs.")

    sel_idx = np.concatenate([pos_idx, neg_idx])
    y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))]).astype(np.float32)

    # shuffle
    perm = rng.permutation(len(sel_idx))
    sel_idx = sel_idx[perm]
    y = y[perm]
    X_sel = X_all[sel_idx]

    print(f"Self-train set: N={len(X_sel)} (pos={int(y.sum())}, neg={int((1-y).sum())})")

    # dataloader
    ds = Seq2PointDataset(X_sel, y)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)

    # load pretrained weights (if finetuned model exists prefer that? here we use MODEL_IN)
    sd = torch.load(MODEL_IN, map_location=device)
    model.load_state_dict(sd)

    # loss (balanced)
    pos = float(y.sum()); neg = float(len(y) - pos)
    pos_weight = torch.tensor([neg / (pos + 1e-6)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        print(f"Epoch {epoch:02d} | loss={tot/len(ds):.4f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print("Saved:", MODEL_OUT)

if __name__ == "__main__":
    main()
