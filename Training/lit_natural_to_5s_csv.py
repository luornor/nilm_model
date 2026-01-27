import os, glob
import numpy as np
import pandas as pd
from scipy.io import loadmat

IN_ROOT = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Matlab_Data\Natural"
OUT_CSV = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\lit_natural_5s.csv"
BIN_S = 5

rows = []
for fp in glob.glob(os.path.join(IN_ROOT, "**", "Waveform*.mat"), recursive=True):
    S = loadmat(fp, squeeze_me=True)
    if "vGrid" not in S or "iHall" not in S or "sps" not in S:
        continue
    v = S["vGrid"].astype(np.float32).reshape(-1)
    i = S["iHall"].astype(np.float32).reshape(-1)
    p = v * i
    sps = float(np.array(S["sps"]).reshape(()))

    binN = int(round(sps * BIN_S))
    K = len(p) // binN
    if K < 2:
        continue

    P = p[:K*binN].reshape(K, binN).mean(axis=1)
    t_sec = np.arange(K) * BIN_S
    file_rel = fp.replace(r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset" + "\\", "")

    for t, val in zip(t_sec, P):
        rows.append((file_rel, int(t), float(val)))

df = pd.DataFrame(rows, columns=["file", "t_sec", "P"])
df.to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV, "rows:", len(df))
