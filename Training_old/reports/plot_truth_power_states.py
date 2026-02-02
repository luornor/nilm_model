import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\lit_synth_5s_states.csv"
OUT_PNG  = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\plot_truth_power_states.png"

# pick one file that actually contains 2 loads A0+B0 (change if needed)
TARGET_FILE_CONTAINS = r"2A0B0"

df = pd.read_csv(CSV_PATH)

# pick a file path that contains your target acquisition
cand = df[df["file"].str.contains(TARGET_FILE_CONTAINS, regex=False)]["file"].unique()
if len(cand) == 0:
    raise RuntimeError(f"No files found containing: {TARGET_FILE_CONTAINS}. Change TARGET_FILE_CONTAINS.")
file_id = cand[0]

g = df[df["file"] == file_id].sort_values("t_sec").reset_index(drop=True)

plt.figure(figsize=(10,4))
plt.plot(g["t_sec"], g["P"])
plt.ylabel("Total Power (LF)")
plt.xlabel("Time (s)")
plt.title("Total Power (blue) + Appliance ON/OFF (step lines)")

# plot state lines scaled to sit at the bottom (simple and readable)
base = g["P"].min()
rng = max(g["P"].max() - base, 1e-6)

for col in ["y_A0", "y_B0"]:
    if col in g.columns:
        plt.step(g["t_sec"], base + 0.10*rng*g[col], where="post", linewidth=2, label=col)

plt.legend()
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print("Saved:", OUT_PNG)
