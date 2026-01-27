import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

# natural data inference on synthetic-trained models

NAT_CSV = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\dataset_undersampled_5s\lit_natural_5s.csv"
MODEL_DIR = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\models_trained_on_synth_sim_data"

OUT_PRED = os.path.join(MODEL_DIR, "natural_predictions.csv")

TOP_APPLIANCES = [
    "y_Incandescent_Lamp_N0",
    "y_AC_Adapter_Sony_M0",
    "y_Oil_Heater_Q0",
    "y_Soldering_Station_H0",
    "y_Smoke_Extractor_E0",
    "y_Phone_Charger_Motorola_I0",
]
   # best from your results
WINDOW = 5

#probabilities per appliance
""" Predict on natural aggregate data using trained models from synthetic and simulated data."""
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 1))

    def forward(self, x): return self.head(self.net(x))

def make_windows_p(p, window):
    X, centers = [], []
    for i in range(0, len(p) - window + 1):
        w = p[i:i+window].copy()
        w = (w - w.mean()) / (w.std() + 1e-6)
        X.append(w)
        centers.append(i + window//2)
    return np.stack(X), np.array(centers)

df = pd.read_csv(NAT_CSV)
out_rows = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for file_id, g in df.groupby("file"):
    g = g.sort_values("t_sec").reset_index(drop=True)
    p = g["P"].to_numpy(np.float32)

    if len(p) < WINDOW:
        continue

    X, centers = make_windows_p(p, WINDOW)
    X_t = torch.tensor(X, dtype=torch.float32)[:, None, :].to(device)
    # the p is the power signal from natural dataset
    pred = {"file": file_id, "t_sec": g["t_sec"].tolist(), "P": g["P"].tolist()}

    for target in TOP_APPLIANCES:
        base = os.path.join(MODEL_DIR, f"cnn_seq2point_{target}.pt")
        ft   = os.path.join(MODEL_DIR, f"cnn_seq2point_{target}_finetuned_natural.pt")

        model_path = ft if os.path.isfile(ft) else base
        if not os.path.isfile(model_path):
            print("Missing:", base, "and", ft)
            continue

        model = SmallCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            probs = torch.sigmoid(model(X_t)).cpu().numpy().reshape(-1)

        prob_full = np.full(len(p), np.nan, dtype=np.float32)
        prob_full[centers] = probs
        pred[f"prob_{target}"] = prob_full.tolist()

        print(f"Loaded {target} from:", os.path.basename(model_path))


    out_rows.append(pd.DataFrame(pred))

pred_df = pd.concat(out_rows, ignore_index=True)
pred_df.to_csv(OUT_PRED, index=False)
print("Wrote:", OUT_PRED)

# quick plots (first file only)
PLOT_DIR = MODEL_DIR

for target in TOP_APPLIANCES:
    col = f"prob_{target}"
    if col not in pred_df.columns:
        continue

    x = pred_df[col].to_numpy(dtype=np.float32)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        continue

    plt.figure(figsize=(7,4))
    plt.hist(x, bins=30)
    plt.title(f"Confidence histogram ({target})")
    plt.xlabel("Model confidence (0–1)")
    plt.ylabel("Count")
    out_png = os.path.join(PLOT_DIR, f"hist_{target}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print("Saved:", out_png)

# simple ON/OFF detection with hysteresis
def moving_average(a, k=5):
    a = np.asarray(a, dtype=np.float32)
    out = np.full_like(a, np.nan)
    valid = ~np.isnan(a)
    if valid.sum() < k:
        return out
    av = np.convolve(a[valid], np.ones(k)/k, mode="same")
    out[valid] = av
    return out

POS_THR = 0.55
NEG_THR = 0.45
SMOOTH_K = 5   # 5 points * 5s = 25s smoothing

# 3rd plot: power + smoothed confidence + ON/OFF for each top appliance
one_file = pred_df["file"].iloc[0]
g = pred_df[pred_df["file"] == one_file].sort_values("t_sec")

for target in TOP_APPLIANCES:
    col = f"prob_{target}"
    if col not in g.columns:
        continue

    prob = g[col].to_numpy(np.float32)
    prob_s = moving_average(prob, k=SMOOTH_K)

    # simple hysteresis ON/OFF
    on = np.zeros(len(prob_s), dtype=np.int32)
    state = 0
    for i, p in enumerate(prob_s):
        if np.isnan(p):
            on[i] = state
            continue
        if state == 0 and p >= POS_THR:
            state = 1
        elif state == 1 and p <= NEG_THR:
            state = 0
        on[i] = state

    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(g["t_sec"], g["P"])
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Total Power (LF)")

    ax2 = ax1.twinx()
    ax2.plot(g["t_sec"], prob_s)
    ax2.plot(g["t_sec"], on)  # ON/OFF line
    ax2.set_ylabel("Confidence / ON-OFF")
    ax2.set_ylim(0, 1)

    plt.title(f"Natural: power + smoothed confidence + ON/OFF ({target})")
    plt.tight_layout()
    out_png = os.path.join(MODEL_DIR, f"natural_{target}_onoff.png")
    plt.savefig(out_png, dpi=200)
    print("Saved:", out_png)


one_file = pred_df["file"].iloc[0]
g = pred_df[pred_df["file"] == one_file].sort_values("t_sec")

plt.figure(figsize=(12,5))
plt.plot(g["t_sec"], g["P"], label="Total Power")

for target in TOP_APPLIANCES:
    col = f"prob_{target}"
    if col not in g.columns:
        continue
    prob = g[col].to_numpy(np.float32)
    prob = np.nan_to_num(prob, nan=0.0)
    plt.plot(g["t_sec"], prob, label=target)

plt.title("Natural: total power + model confidences (top appliances)")
plt.xlabel("Time (s)")
plt.ylabel("Power / Confidence")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
out_png = os.path.join(MODEL_DIR, "natural_all_top6_overlay.png")
plt.savefig(out_png, dpi=200)
print("Saved:", out_png)
