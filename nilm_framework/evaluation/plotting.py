"""Plotting utilities for NILM results."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


def moving_average(a: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Compute moving average with NaN handling.
    
    Args:
        a: Input array
        k: Window size
        
    Returns:
        Smoothed array
    """
    a = np.asarray(a, dtype=np.float32)
    out = np.full_like(a, np.nan)
    valid = ~np.isnan(a)
    
    if valid.sum() < k:
        return out
    
    av = np.convolve(a[valid], np.ones(k) / k, mode="same")
    out[valid] = av
    return out


def plot_confidence_histogram(
    probabilities: np.ndarray,
    title: str = "Confidence Histogram",
    output_path: Optional[str] = None,
    bins: int = 30,
):
    """
    Plot histogram of prediction probabilities.
    
    Args:
        probabilities: Array of probabilities
        title: Plot title
        output_path: Path to save plot (optional)
        bins: Number of histogram bins
    """
    probs_clean = probabilities[~np.isnan(probabilities)]

    if len(probs_clean) == 0:
        print("No valid probabilities to plot")
        return

    plt.figure(figsize=(7, 4))

    # Newer NumPy versions can raise an error if the data range is
    # extremely small relative to the requested number of bins.
    # Try the requested bin count first, and if it fails, fall back
    # to a single-bin histogram so plotting never crashes inference.
    try:
        plt.hist(probs_clean, bins=bins)
    except ValueError:
        plt.hist(probs_clean, bins=1)
    plt.title(title)
    plt.xlabel("Model Confidence (0–1)")
    plt.ylabel("Count")
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=200)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_onoff_states(
    df: pd.DataFrame,
    appliance: str,
    prob_col: str,
    on_threshold: float = 0.55,
    off_threshold: float = 0.45,
    smoothing_window: int = 5,
    output_path: Optional[str] = None,
):
    """
    Plot power signal with confidence and ON/OFF states.
    
    Args:
        df: DataFrame with 't_sec', 'P', and prob_col columns
        appliance: Appliance name (for title)
        prob_col: Column name for probabilities
        on_threshold: Threshold for turning ON
        off_threshold: Threshold for turning OFF
        smoothing_window: Window size for moving average
        output_path: Path to save plot (optional)
    """
    if prob_col not in df.columns:
        print(f"Column {prob_col} not found")
        return
    
    prob = df[prob_col].to_numpy(dtype=np.float32)
    prob_smooth = moving_average(prob, k=smoothing_window)
    
    # Hysteresis ON/OFF detection
    on_states = np.zeros(len(prob_smooth), dtype=np.int32)
    state = 0
    
    for i, p in enumerate(prob_smooth):
        if np.isnan(p):
            on_states[i] = state
            continue
        if state == 0 and p >= on_threshold:
            state = 1
        elif state == 1 and p <= off_threshold:
            state = 0
        on_states[i] = state
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    ax1.plot(df["t_sec"], df["P"], label="Total Power", color="blue")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Total Power (W)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    
    ax2 = ax1.twinx()
    ax2.plot(df["t_sec"], prob_smooth, label="Confidence", color="orange", alpha=0.7)
    ax2.plot(df["t_sec"], on_states, label="ON/OFF", color="red", linewidth=2)
    ax2.set_ylabel("Confidence / ON-OFF", color="red")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor="red")
    
    plt.title(f"Power + Confidence + ON/OFF States ({appliance})")
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=200)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_overlay(
    df: pd.DataFrame,
    appliances: list,
    output_path: Optional[str] = None,
):
    """
    Plot overlay of total power and all appliance confidences.
    
    Args:
        df: DataFrame with 't_sec', 'P', and prob_* columns
        appliances: List of appliance names
        output_path: Path to save plot (optional)
    """
    plt.figure(figsize=(12, 5))
    plt.plot(df["t_sec"], df["P"], label="Total Power", linewidth=2, color="black")
    
    for appliance in appliances:
        prob_col = f"prob_{appliance}"
        if prob_col not in df.columns:
            continue
        
        prob = df[prob_col].to_numpy(dtype=np.float32)
        prob = np.nan_to_num(prob, nan=0.0)
        plt.plot(df["t_sec"], prob, label=appliance, alpha=0.7)
    
    plt.title("Total Power + Model Confidences (All Appliances)")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W) / Confidence")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=200)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()
