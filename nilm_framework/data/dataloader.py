"""DataLoader creation utilities."""

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
from .dataset import create_dataset_from_dataframe


def create_dataloaders(
    df: pd.DataFrame,
    target_col: str,
    window_size: int = 5,
    batch_size: int = 256,
    val_split: float = 0.2,
    stratify: bool = True,
    shuffle_train: bool = True,
    normalize: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from DataFrame.
    
    Args:
        df: DataFrame with 'file', 'P', and target_col columns
        target_col: Target column name
        window_size: Window size in bins
        batch_size: Batch size
        val_split: Validation split ratio
        stratify: Whether to stratify split by target
        shuffle_train: Whether to shuffle training data
        normalize: Whether to normalize windows
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Filter valid rows
    df_clean = df.dropna(subset=["P", target_col, "file"])
    
    if len(df_clean) == 0:
        raise ValueError(f"No valid rows for target {target_col}")
    
    # Create windows for all files
    X_list = []
    y_list = []
    file_ids = []
    
    for file_id, group in df_clean.groupby("file"):
        if "t_sec" in group.columns:
            group = group.sort_values("t_sec")
        
        power = group["P"].to_numpy(dtype=np.float32)
        target = group[target_col].to_numpy(dtype=np.float32)
        
        # Create windows
        from ..utils.data_utils import make_windows
        X, y = make_windows(power, window_size, target=target, normalize=normalize)
        
        if len(X) > 0:
            X_list.append(X)
            y_list.append(y)
            file_ids.extend([file_id] * len(X))
    
    if not X_list:
        raise ValueError(f"No windows created for target {target_col}")
    
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # Split
    indices = np.arange(len(X))
    if stratify and len(np.unique(y)) > 1:
        tr_idx, val_idx = train_test_split(
            indices,
            test_size=val_split,
            random_state=seed,
            stratify=y,
        )
    else:
        tr_idx, val_idx = train_test_split(
            indices,
            test_size=val_split,
            random_state=seed,
        )
    
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Create datasets from pre-windowed data
    from .dataset import NILMDataset
    
    train_dataset = NILMDataset.from_windows(X_tr, y_tr)
    val_dataset = NILMDataset.from_windows(X_val, y_val)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    return train_loader, val_loader
