"""Data preprocessing and utility functions."""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def normalize_window(window: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize a window by subtracting mean and dividing by std.
    
    Args:
        window: Power window array
        eps: Small value to avoid division by zero
        
    Returns:
        Normalized window
    """
    mean = window.mean()
    std = window.std()
    if std < eps:
        return window - mean
    return (window - mean) / std


def make_windows(
    power: np.ndarray,
    window_size: int,
    target: Optional[np.ndarray] = None,
    normalize: bool = True,
    center_index: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Create sliding windows from power signal.
    
    Args:
        power: Power signal array (1D)
        window_size: Number of bins in window
        target: Optional target array (same length as power)
        normalize: Whether to normalize each window
        center_index: Index to use for target alignment (default: window_size // 2)
        
    Returns:
        Tuple of (X, y) where:
        - X: Windowed features (N, window_size)
        - y: Target values at center points (N,) or None
    """
    if center_index is None:
        center_index = window_size // 2
    
    if len(power) < window_size:
        X = np.zeros((0, window_size), dtype=np.float32)
        y = np.array([], dtype=np.float32) if target is not None else None
        return X, y
    
    X_list = []
    y_list = [] if target is not None else None
    
    for i in range(len(power) - window_size + 1):
        window = power[i:i + window_size].copy().astype(np.float32)
        
        if normalize:
            window = normalize_window(window)
        
        X_list.append(window)
        
        if target is not None:
            center_idx = i + center_index
            if center_idx < len(target):
                y_list.append(target[center_idx])
    
    X = np.stack(X_list) if X_list else np.zeros((0, window_size), dtype=np.float32)
    
    if y_list is not None:
        y = np.array(y_list, dtype=np.float32)
    else:
        y = None
    
    return X, y


def find_appliance_columns(df: pd.DataFrame) -> List[str]:
    """
    Find all appliance target columns (columns starting with 'y_').
    
    Args:
        df: DataFrame to search
        
    Returns:
        List of column names
    """
    return [col for col in df.columns if col.startswith("y_")]


def load_dataframe(csv_path: str, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load CSV file and validate required columns.
    
    Args:
        csv_path: Path to CSV file
        required_columns: List of required column names
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    df = pd.read_csv(csv_path)
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    return df
