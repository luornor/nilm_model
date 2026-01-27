import pandas as pd
import matplotlib.pyplot as plt

RES_PATH = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\synthetic_cnn_results.csv"
OUT_PNG  = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Training\reports\plot_f1_all23.png"

# Load and sort all models by F1
res = pd.read_csv(RES_PATH).sort_values("F1", ascending=False)

plt.figure(figsize=(14, 5))
plt.bar(res["target"], res["F1"])   # <-- model = column with model names
plt.ylim(0, 1)
plt.ylabel("F1 (0=bad, 1=perfect)")
plt.title("F1 Score for All 23 Models (Synthetic, 5s)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)

print("Saved:", OUT_PNG)
