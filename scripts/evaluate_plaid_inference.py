#!/usr/bin/env python
"""Evaluate PLAID models by comparing predictions to labels.

This script joins:
- Ground truth labels from Exports/plaid_train_1s.csv
- Predictions from outputs/plaid_inference/plaid_preds_1s.csv

and computes metrics per appliance to verify that:
- The labelling logic is sensible
- The models trained on PLAID_Data actually match those labels.

Usage (from project root):

    py -3.11 scripts/evaluate_plaid_inference.py \
        --labels Exports/plaid_train_1s.csv \
        --preds outputs/plaid_inference/plaid_preds_1s.csv \
        --output outputs/plaid_inference/plaid_eval_1s.csv

"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `nilm_framework` can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from nilm_framework.training.metrics import compute_metrics


def get_common_appliances(label_cols: List[str], pred_cols: List[str]) -> List[str]:
    """Find appliances that exist in both labels (y_*) and preds (prob_y_*)."""

    label_apps = [c for c in label_cols if c.startswith("y_")]
    pred_apps = [c[len("prob_") :] for c in pred_cols if c.startswith("prob_")]
    common = sorted(set(label_apps) & set(pred_apps))
    return common


def evaluate(
    labels_path: Path,
    preds_path: Path,
    output_path: Path,
    threshold: float = 0.5,
) -> None:
    """Evaluate predictions against labels and save per-appliance metrics."""

    print(f"Loading labels from: {labels_path}")
    df_labels = pd.read_csv(labels_path)

    print(f"Loading predictions from: {preds_path}")
    df_preds = pd.read_csv(preds_path)

    # Join on file and t_sec to align rows
    merge_keys = ["file", "t_sec"]
    missing_keys = [k for k in merge_keys if k not in df_labels.columns or k not in df_preds.columns]
    if missing_keys:
        raise ValueError(f"Missing join keys in CSVs: {missing_keys}")

    df = pd.merge(df_labels, df_preds, on=merge_keys, how="inner", suffixes=("_label", "_pred"))
    print(f"Merged rows: {len(df)}")

    label_cols = df_labels.columns.tolist()
    pred_cols = df_preds.columns.tolist()
    appliances = get_common_appliances(label_cols, pred_cols)

    if not appliances:
        raise ValueError("No common appliances between labels and predictions.")

    print(f"Found {len(appliances)} common appliances:")
    for a in appliances:
        print(f"  - {a}")

    rows = []
    for appliance in appliances:
        y_col = appliance
        prob_col = f"prob_{appliance}"

        y_true = df[y_col].to_numpy(dtype=np.float32)
        y_prob = df[prob_col].to_numpy(dtype=np.float32)

        # Drop rows where either the label or probability is NaN.
        # NaN labels appear for appliances that are not present in a
        # particular scenario; they should not be counted as 0 or 1.
        mask = (~np.isnan(y_true)) & (~np.isnan(y_prob))
        y_true_clean = y_true[mask]
        y_prob_clean = y_prob[mask]

        if len(y_true_clean) == 0:
            continue

        metrics = compute_metrics(y_true_clean, y_pred=y_prob_clean, y_prob=y_prob_clean, threshold=threshold)
        metrics_row = {"appliance": appliance}
        metrics_row.update(metrics)
        rows.append(metrics_row)

    if not rows:
        raise RuntimeError("No metrics computed; check data and columns.")

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values("f1", ascending=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Saved evaluation metrics to: {output_path}")

    print("\nPer-appliance metrics (top to bottom by F1):")
    print(df_out[["appliance", "f1", "precision", "recall", "accuracy", "pos_rate", "pred_pos_rate"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PLAID predictions against PLAID labels")
    parser.add_argument("--labels", type=str, required=True, help="Path to plaid_train_1s.csv (with y_* labels)")
    parser.add_argument("--preds", type=str, required=True, help="Path to plaid_preds_1s.csv (with prob_y_* columns)")
    parser.add_argument("--output", type=str, required=True, help="Where to save evaluation CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for ON/OFF (default 0.5)")

    args = parser.parse_args()

    evaluate(
        labels_path=Path(args.labels),
        preds_path=Path(args.preds),
        output_path=Path(args.output),
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
