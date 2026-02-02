import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


"""Preprocessing for Dataset/David_Data.

This script:
- Reads high-rate aggregated CSVs (30 kHz, 10 s) and metadata_aggregated.json.
- Downsamples to 1 s bins using mean power.
- Uses metadata ON/OFF sample indices to create binary ON/OFF labels per appliance.
- Writes a unified training CSV compatible with the NILM framework.

Resulting CSV format (example):
    file,t_sec,P,y_Fan,y_Vacuum,...

Usage (from repo root):
    python -m scripts.preprocess_data \
        --data-root Dataset/David_Data \
        --output-csv Exports/david_train_1s.csv

You can adjust bin_length_sec (default 1.0) or filtering logic as needed.
"""


def parse_sample_index(value: str) -> int:
    """Parse an index string like "[37043]" -> 37043.

    If the string is empty or malformed, returns -1.
    """

    if not value:
        return -1
    s = value.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    try:
        return int(s)
    except ValueError:
        return -1


def build_appliance_name(appliance: Dict) -> str:
    """Build a stable column name for an appliance.

    We use a simple scheme based on the `type` field, optionally with brand
    if there are duplicates. The caller can post-process or map these names
    to the framework's y_* convention if desired.
    """

    app_type = appliance.get("type", "Unknown").strip()
    # Normalize spaces and case
    app_type_clean = app_type.replace(" ", "_")
    return f"y_{app_type_clean}"


def downsample_aggregate(
    agg_samples: np.ndarray,
    fs: float,
    bin_length_sec: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample high-rate aggregate samples into mean power per bin.

    Returns (t_sec, P), where:
    - t_sec: center time of each bin in seconds.
    - P: mean value of samples within each bin.
    """

    n_samples = len(agg_samples)
    samples_per_bin = int(round(fs * bin_length_sec))
    if samples_per_bin <= 0:
        raise ValueError("samples_per_bin must be positive")

    n_bins = n_samples // samples_per_bin
    if n_bins == 0:
        return np.array([]), np.array([])

    agg_samples = agg_samples[: n_bins * samples_per_bin]
    P = agg_samples.reshape(n_bins, samples_per_bin).mean(axis=1)
    # Bin centers: (i + 0.5) * bin_length_sec
    t_sec = (np.arange(n_bins, dtype=float) + 0.5) * bin_length_sec
    return t_sec, P


def build_labels_for_appliances(
    appliances: List[Dict],
    fs: float,
    t_sec: np.ndarray,
    bin_length_sec: float,
) -> Dict[str, np.ndarray]:
    """Create binary ON/OFF labels per appliance for each time bin.

    Strategy:
    - Convert on/off sample indices to times.
    - For each bin, mark ON if more than 50% of the underlying samples in
      that bin lie between on and off times.

    For now we assume a single contiguous on->off interval per appliance.
    If on or off is missing or invalid, that appliance is labelled all zeros.
    """

    labels: Dict[str, np.ndarray] = {}
    n_bins = len(t_sec)
    half_bin = 0.5 * bin_length_sec

    for appliance in appliances:
        col_name = build_appliance_name(appliance)
        on_idx = parse_sample_index(appliance.get("on", ""))
        off_idx = parse_sample_index(appliance.get("off", ""))

        if on_idx < 0 or off_idx <= on_idx:
            labels[col_name] = np.zeros(n_bins, dtype=int)
            continue

        t_on = on_idx / fs
        t_off = off_idx / fs

        # Fraction-based labelling: ON if >50% of bin duration within [t_on, t_off]
        y = np.zeros(n_bins, dtype=int)
        for i, center in enumerate(t_sec):
            bin_start = center - half_bin
            bin_end = center + half_bin
            overlap_start = max(bin_start, t_on)
            overlap_end = min(bin_end, t_off)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap >= 0.5 * bin_length_sec:
                y[i] = 1
        labels[col_name] = y

    return labels


def load_aggregate_csv(path: Path) -> np.ndarray:
    """Load aggregated CSV and return a 1D numpy array of the first column.

    David_Data aggregated files appear to have two numeric columns without
    a header. We treat the first column as the aggregate signal.
    """

    df = pd.read_csv(path, header=None)
    # Use the first column as aggregate; adjust here if needed
    return df.iloc[:, 0].to_numpy(dtype=float)


def preprocess_data(
    data_root: Path,
    output_csv: Path,
    bin_length_sec: float = 1.0,
    fs_default: float = 30000.0,
) -> None:
    """Main preprocessing routine for David_Data.

    Parameters
    ----------
    data_root : Path
        Path to Dataset/David_Data.
    output_csv : Path
        Where to write the unified training CSV.
    bin_length_sec : float
        Length of each time bin in seconds (default 1.0).
    fs_default : float
        Default sampling frequency if not specified in metadata.
    """

    metadata_path = data_root / "metadata_aggregated.json"
    aggregated_dir = data_root / "aggregated"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata_aggregated.json not found at {metadata_path}")
    if not aggregated_dir.is_dir():
        raise FileNotFoundError(f"aggregated directory not found at {aggregated_dir}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    rows: List[Dict] = []

    for key, info in metadata.items():
        # Each key corresponds to an aggregated/<key>.csv file
        agg_path = aggregated_dir / f"{key}.csv"
        if not agg_path.is_file():
            continue

        appliances = info.get("appliances", [])
        header = info.get("header", {})
        fs_str = header.get("sampling_frequency", "")
        fs = fs_default
        if isinstance(fs_str, str) and fs_str.endswith("Hz"):
            try:
                fs = float(fs_str[:-2])
            except ValueError:
                fs = fs_default

        agg_samples = load_aggregate_csv(agg_path)
        t_sec, P = downsample_aggregate(agg_samples, fs=fs, bin_length_sec=bin_length_sec)
        if len(t_sec) == 0:
            continue

        label_dict = build_labels_for_appliances(
            appliances=appliances,
            fs=fs,
            t_sec=t_sec,
            bin_length_sec=bin_length_sec,
        )

        # Ensure stable column ordering: sort appliance columns by name
        label_cols = sorted(label_dict.keys())

        for i in range(len(t_sec)):
            row: Dict = {
                "file": str(key),
                "t_sec": float(t_sec[i]),
                "P": float(P[i]),
            }
            for col in label_cols:
                row[col] = int(label_dict[col][i])
            rows.append(row)

    if not rows:
        raise RuntimeError("No data rows produced; check metadata and aggregated CSVs.")

    df_out = pd.DataFrame(rows)
    # Sort columns: file, t_sec, P, then y_*
    base_cols = ["file", "t_sec", "P"]
    y_cols = sorted([c for c in df_out.columns if c.startswith("y_")])
    df_out = df_out[base_cols + y_cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Wrote {len(df_out)} rows to {output_csv}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess David_Data into training CSV")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path("Dataset") / "David_Data"),
        help="Path to Dataset/David_Data",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(Path("Exports") / "david_train_1s.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--bin-length-sec",
        type=float,
        default=1.0,
        help="Length of each time bin in seconds (default 1.0)",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_csv = Path(args.output_csv)

    preprocess_data(
        data_root=data_root,
        output_csv=output_csv,
        bin_length_sec=args.bin_length_sec,
    )


if __name__ == "__main__":
    main()
