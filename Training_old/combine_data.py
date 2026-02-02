import os
import pandas as pd

SYN = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\dataset_undersampled_5s\lit_synth_5s_states_named.csv"
SIM = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\dataset_undersampled_5s\simulated_data_5s_states.csv"
OUT = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\train_mix_5s.csv"

df_syn = pd.read_csv(SYN)
df_sim = pd.read_csv(SIM)

# Add source tag (helps debugging / split)
df_syn["source"] = "synthetic"
df_sim["source"] = "simulated"

# Union of columns
all_cols = sorted(set(df_syn.columns) | set(df_sim.columns))

# Ensure both have all columns (missing -> 0 for y_*, NaN for P/t_sec/file)
def align(df):
    for c in all_cols:
        if c not in df.columns:
            if c.startswith("y_"):
                df[c] = 0
            else:
                df[c] = pd.NA
    return df[all_cols]

df_syn = align(df_syn)
df_sim = align(df_sim)

# Drop rows missing core fields
core = ["file", "t_sec", "P"]
df_syn = df_syn.dropna(subset=core)
df_sim = df_sim.dropna(subset=core)

df_mix = pd.concat([df_syn, df_sim], ignore_index=True)

# Optional: ensure labels are ints
y_cols = [c for c in df_mix.columns if c.startswith("y_")]
df_mix[y_cols] = df_mix[y_cols].fillna(0).astype(int)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_mix.to_csv(OUT, index=False)

print("Wrote:", OUT)
print("Rows:", len(df_mix), "Cols:", df_mix.shape[1])
print("Synthetic rows:", len(df_syn), "Simulated rows:", len(df_sim))
print("Num labels:", len(y_cols))
