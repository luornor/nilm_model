"""
Inference and prediction utilities for NILM models.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import List, Optional, Dict

from ..data.dataset import NILMDataset
from ..utils.data_utils import make_windows


class Predictor:
    """
    Predictor for running inference on natural data.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained model
            device: Device to use
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict_file(
        self,
        power: np.ndarray,
        window_size: int = 5,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Predict probabilities for a single file's power signal.
        
        Args:
            power: Power signal (1D array)
            window_size: Window size
            batch_size: Batch size for inference
            
        Returns:
            Array of probabilities (same length as power, with NaNs at edges)
        """
        if len(power) < window_size:
            return np.full(len(power), np.nan, dtype=np.float32)
        
        # Create windows
        X, _ = make_windows(power, window_size, normalize=True)
        
        if len(X) == 0:
            return np.full(len(power), np.nan, dtype=np.float32)
        
        # Create dataset from pre-windowed data
        dataset = NILMDataset.from_windows(X, y=None)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Predict
        all_probs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                logits = self.model(x)
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
                all_probs.extend(probs)
        
        # Map back to original signal length
        probs_full = np.full(len(power), np.nan, dtype=np.float32)
        center = window_size // 2
        probs_full[center : center + len(all_probs)] = all_probs
        
        return probs_full
    
    def predict_dataframe(
        self,
        df: pd.DataFrame,
        window_size: int = 5,
        batch_size: int = 256,
    ) -> pd.DataFrame:
        """
        Predict probabilities for all files in DataFrame.
        
        Args:
            df: DataFrame with 'file', 'P', and optionally 't_sec' columns
            window_size: Window size
            batch_size: Batch size
            
        Returns:
            DataFrame with added probability column
        """
        result_rows = []
        
        for file_id, group in df.groupby("file"):
            if "t_sec" in group.columns:
                group = group.sort_values("t_sec")
            
            power = group["P"].to_numpy(dtype=np.float32)
            
            # Predict
            probs = self.predict_file(power, window_size, batch_size)
            
            # Build result row
            row = {
                "file": file_id,
                "t_sec": group["t_sec"].tolist() if "t_sec" in group.columns else None,
                "P": group["P"].tolist(),
                "prob": probs.tolist(),
            }
            result_rows.append(pd.DataFrame(row))
        
        result_df = pd.concat(result_rows, ignore_index=True)
        return result_df
    
    @staticmethod
    def predict_multiple_appliances(
        models: Dict[str, torch.nn.Module],
        df: pd.DataFrame,
        window_size: int = 5,
        batch_size: int = 256,
        device: Optional[torch.device] = None,
    ) -> pd.DataFrame:
        """
        Predict probabilities for multiple appliances.
        
        Args:
            models: Dictionary mapping appliance names to models
            df: DataFrame with 'file', 'P' columns
            window_size: Window size
            batch_size: Batch size
            device: Device to use
            
        Returns:
            DataFrame with probability columns for each appliance
        """
        result_rows = []
        
        for file_id, group in df.groupby("file"):
            if "t_sec" in group.columns:
                group = group.sort_values("t_sec")
            
            power = group["P"].to_numpy(dtype=np.float32)
            
            # Build result row
            row = {
                "file": file_id,
                "t_sec": group["t_sec"].tolist() if "t_sec" in group.columns else None,
                "P": group["P"].tolist(),
            }
            
            # Predict for each appliance
            for appliance, model in models.items():
                predictor = Predictor(model, device=device)
                probs = predictor.predict_file(power, window_size, batch_size)
                row[f"prob_{appliance}"] = probs.tolist()
            
            result_rows.append(pd.DataFrame(row))
        
        result_df = pd.concat(result_rows, ignore_index=True)
        return result_df
