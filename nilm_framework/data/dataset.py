"""
Dataset classes for NILM training and inference.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
from ..utils.data_utils import make_windows, find_appliance_columns


class NILMDataset(Dataset):
    """
    Base dataset for NILM data.
    
    Handles windowing and normalization of power signals.
    """
    
    def __init__(
        self,
        power: np.ndarray,
        target: Optional[np.ndarray] = None,
        window_size: int = 5,
        normalize: bool = True,
        center_index: Optional[int] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            power: Power signal (1D array)
            target: Target values (1D array, same length as power)
            window_size: Number of bins in sliding window
            normalize: Whether to normalize each window
            center_index: Index for target alignment (default: window_size // 2)
        """
        if center_index is None:
            center_index = window_size // 2
        
        X, y = make_windows(
            power,
            window_size,
            target=target,
            normalize=normalize,
            center_index=center_index,
        )
        
        # Convert to tensors
        self.X = torch.tensor(X, dtype=torch.float32)[:, None, :]  # (N, 1, T)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)[:, None]  # (N, 1)
        else:
            self.y = None
    
    @classmethod
    def from_windows(
        cls,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> "NILMDataset":
        """
        Create dataset from pre-windowed data.
        
        Args:
            X: Windowed features (N, window_size)
            y: Target values (N,) or None
            
        Returns:
            NILMDataset instance
        """
        dataset = cls.__new__(cls)
        dataset.X = torch.tensor(X, dtype=torch.float32)[:, None, :]  # (N, 1, T)
        if y is not None:
            dataset.y = torch.tensor(y, dtype=torch.float32)[:, None]  # (N, 1)
        else:
            dataset.y = None
        return dataset
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx], None


class Seq2PointDataset(NILMDataset):
    """
    Seq2Point dataset - predicts center point of window.
    
    This is an alias for NILMDataset with the standard Seq2Point configuration.
    """
    
    def __init__(
        self,
        power: np.ndarray,
        target: Optional[np.ndarray] = None,
        window_size: int = 5,
        normalize: bool = True,
    ):
        super().__init__(
            power=power,
            target=target,
            window_size=window_size,
            normalize=normalize,
            center_index=window_size // 2,
        )


def create_dataset_from_dataframe(
    df: pd.DataFrame,
    target_col: str,
    window_size: int = 5,
    normalize: bool = True,
    group_by_file: bool = True,
) -> NILMDataset:
    """
    Create dataset from DataFrame with multiple files.
    
    Args:
        df: DataFrame with columns: 'file', 'P', and target_col
        target_col: Name of target column
        window_size: Window size in bins
        normalize: Whether to normalize windows
        group_by_file: Whether to group by 'file' column
        
    Returns:
        Combined dataset from all files
    """
    X_list = []
    y_list = []
    
    if group_by_file and "file" in df.columns:
        groups = df.groupby("file")
    else:
        groups = [("all", df)]
    
    for file_id, group in groups:
        # Sort by time if available
        if "t_sec" in group.columns:
            group = group.sort_values("t_sec")
        
        # Extract power and target
        power = group["P"].to_numpy(dtype=np.float32)
        
        if target_col in group.columns:
            target = group[target_col].to_numpy(dtype=np.float32)
        else:
            target = None
        
        # Create windows
        X, y = make_windows(
            power,
            window_size,
            target=target,
            normalize=normalize,
        )
        
        if len(X) > 0:
            X_list.append(X)
            if y is not None:
                y_list.append(y)
    
    if not X_list:
        # Return empty dataset
        X = np.zeros((0, window_size), dtype=np.float32)
        y = np.zeros((0,), dtype=np.float32) if target_col else None
    else:
        X = np.concatenate(X_list, axis=0)
        if y_list:
            y = np.concatenate(y_list, axis=0)
        else:
            y = None
    
    # Create dataset from pre-windowed data
    return NILMDataset.from_windows(X, y)
