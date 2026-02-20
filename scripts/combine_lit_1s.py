"""
Combine 1s LIT synthetic and simulated exports into a single training CSV.

Outputs: Exports/train_mix_1s.csv

Usage (from project root):
    py -3.11 scripts/combine_lit_1s.py
"""

import re
from pathlib import Path

import pandas as pd

SYNTH_CSV  = Path("Exports/lit_synth_1s_states.csv")
SIM_CSV    = Path("Exports/simulated_data_1s_states.csv")
OUTPUT_CSV = Path("Exports/lit_train_mix_1s.csv")

# Columns that are never appliance state columns
NON_APPLIANCE = {"file", "t_sec", "P", "source"}


def _normalize_col(col: str) -> str:
    """
    Normalize verbose simulated column names to the short form used by the
    synthetic export.

    The simulated export produces names like:
        y_AC_Adapter_Sony_PCG_61112L_127V_92W__M0
    The synthetic export (and training framework) uses:
        y_AC_Adapter_Sony_M0

    Strategy: if a y_* column ends with  __XY  (double-underscore + label),
    strip everything back to y_<NiceName>__<Label> → y_<Label-deduced short name>.

    Simpler heuristic: the label suffix after __ is the canonical ID (e.g. M0).
    We look up whether the synthetic CSV already has a column  y_*M0  and if so
    use that name; otherwise we keep the last __ segment as  y_<NiceSafe>_<Label>.
    """
    if not col.startswith("y_"):
        return col
    # Already a clean name (no double underscore)
    if "__" not in col:
        return col
    # Strip the verbose prefix: take everything after the last "__"
    label = col.rsplit("__", 1)[-1]   # e.g. "M0"
    nice  = col[2:].rsplit("__", 1)[0]  # e.g. "AC_Adapter_Sony_PCG_61112L_127V_92W"

    # Simple canonical form: y_<ShortNice>_<Label>
    # We drop the model/voltage spam by keeping only up to the first digit-run
    # that looks like a wattage or voltage (heuristic).
    clean = re.sub(r"_\d+V.*", "", nice)   # remove _127V and everything after
    clean = re.sub(r"_\d+W.*", "", clean)  # remove _92W and everything after
    clean = re.sub(r"_+$", "", clean)      # trailing underscores

    return f"y_{clean}_{label}"


def load_and_tag(path: Path, source_tag: str) -> pd.DataFrame:
    print(f"  Loading {path} …", flush=True)
    df = pd.read_csv(path, low_memory=False)
    df["source"] = source_tag

    # Rename verbose columns on simulated data
    rename = {c: _normalize_col(c) for c in df.columns if c not in NON_APPLIANCE}
    rename = {k: v for k, v in rename.items() if k != v}
    if rename:
        print(f"    Renaming {len(rename)} verbose column(s) to short form")
        df.rename(columns=rename, inplace=True)

    # Deduplicate columns that ended up with the same name after renaming
    dedup = {}
    for col in df.columns:
        if col not in NON_APPLIANCE and col in dedup:
            # Merge: OR of the two binary columns
            df[dedup[col]] = df[[dedup[col], col]].max(axis=1)
            df.drop(columns=[col], inplace=True)
        else:
            dedup[col] = col

    print(f"    {len(df):,} rows, {len(df.columns)} columns")
    return df


def combine(synth_path: Path, sim_path: Path, output_path: Path) -> None:
    print("=== Combining LIT 1-second datasets ===")

    dfs = []
    if synth_path.exists():
        dfs.append(load_and_tag(synth_path,  "synthetic"))
    else:
        print(f"  WARNING: {synth_path} not found — skipping synthetic")

    if sim_path.exists():
        dfs.append(load_and_tag(sim_path, "simulated"))
    else:
        print(f"  WARNING: {sim_path} not found — skipping simulated")

    if not dfs:
        raise FileNotFoundError(
            "Neither synthetic nor simulated 1s CSV found. "
            "Run the MATLAB export scripts first."
        )

    # Union of all columns; fill 0 for missing appliance columns
    print("\nMerging …")
    combined = pd.concat(dfs, ignore_index=True, sort=False)

    app_cols = [c for c in combined.columns if c.startswith("y_")]
    combined[app_cols] = combined[app_cols].fillna(0).astype(int)

    # Canonical column order: file, t_sec, P, source, y_* (sorted)
    y_cols = sorted([c for c in combined.columns if c.startswith("y_")])
    col_order = ["file", "t_sec", "P", "source"] + y_cols
    combined = combined[col_order]

    print(f"\nCombined: {len(combined):,} rows, {len(combined.columns)} columns")
    print(f"  Appliance columns: {len(y_cols)}")
    print(f"  Source breakdown:\n{combined['source'].value_counts().to_string()}")

    # Quick check: every appliance should have at least some positives
    sparse = [c for c in y_cols if combined[c].sum() < 10]
    if sparse:
        print(f"\n  NOTE: {len(sparse)} appliance(s) have <10 positive samples: {sparse}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    combine(SYNTH_CSV, SIM_CSV, OUTPUT_CSV)
