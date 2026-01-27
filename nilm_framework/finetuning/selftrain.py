"""
Self-training / fine-tuning on natural data using high-confidence predictions.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple

from ..config import FineTuningConfig, TrainingConfig
from ..data.dataset import NILMDataset
from ..utils.data_utils import make_windows
from ..training.trainer import Trainer


class SelfTrainer:
    """
    Self-training using high-confidence predictions from a pre-trained model.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: FineTuningConfig,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize self-trainer.
        
        Args:
            model: Pre-trained model
            config: Fine-tuning configuration
            device: Device to use
        """
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def find_probability_column(
        self,
        df: pd.DataFrame,
        model_path: str,
    ) -> str:
        """
        Find probability column in predictions CSV that matches the model.
        
        Args:
            df: DataFrame with prediction columns
            model_path: Path to model file (used to infer appliance name)
            
        Returns:
            Column name for probabilities
        """
        base = os.path.basename(model_path)
        m = re.search(r"(y_[A-Za-z0-9_]+)", base)
        token = m.group(1) if m else None
        
        if not token:
            raise ValueError(f"Could not extract appliance name from {model_path}")
        
        cols = df.columns.tolist()
        candidates = []
        
        # Try exact match or partial match
        token2 = token.replace("y_", "")
        for c in cols:
            cl = c.lower()
            if ("prob" in cl or "pred" in cl or "p_" in cl) and (
                token.lower() in cl or token2.lower() in cl
            ):
                candidates.append(c)
        
        if not candidates:
            # Fallback: any column containing the token
            for c in cols:
                cl = c.lower()
                if token.lower() in cl or token2.lower() in cl:
                    candidates.append(c)
        
        if not candidates:
            raise ValueError(
                f"Could not find probability column for {token}. "
                f"Available columns: {cols[:30]}"
            )
        
        return candidates[0]
    
    def select_confident_samples(
        self,
        probabilities: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select high-confidence positive and negative samples.
        
        Args:
            probabilities: Array of predicted probabilities
            
        Returns:
            Tuple of (selected_indices, labels)
        """
        # High-confidence selection
        pos_idx = np.where(probabilities >= self.config.pos_threshold)[0]
        neg_idx = np.where(probabilities <= self.config.neg_threshold)[0]
        
        print(
            f"Initial selection: pos={len(pos_idx)}, neg={len(neg_idx)} "
            f"(thresholds: pos>={self.config.pos_threshold}, "
            f"neg<={self.config.neg_threshold})"
        )
        
        # Fallback if thresholds too strict
        if (
            self.config.use_fallback_selection
            and (len(pos_idx) < self.config.min_confident_samples
                 or len(neg_idx) < self.config.min_confident_samples)
        ):
            print("Using fallback selection (top-K / bottom-K)")
            kpos = min(
                self.config.max_pos_samples,
                max(len(probabilities) // 10, self.config.min_confident_samples),
            )
            kneg = min(
                self.config.max_neg_samples,
                max(len(probabilities) // 10, self.config.min_confident_samples),
            )
            
            sorted_idx = np.argsort(probabilities)
            neg_idx_fb = sorted_idx[:kneg]
            pos_idx_fb = sorted_idx[-kpos:]
            
            pos_idx = np.unique(np.concatenate([pos_idx, pos_idx_fb]))
            neg_idx = np.unique(np.concatenate([neg_idx, neg_idx_fb]))
            
            print(f"Fallback selection: pos={len(pos_idx)}, neg={len(neg_idx)}")
        
        # Cap samples
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        if len(pos_idx) > self.config.max_pos_samples:
            pos_idx = rng.choice(pos_idx, size=self.config.max_pos_samples, replace=False)
        if len(neg_idx) > self.config.max_neg_samples:
            neg_idx = rng.choice(neg_idx, size=self.config.max_neg_samples, replace=False)
        
        if len(pos_idx) < 50 or len(neg_idx) < 50:
            raise RuntimeError(
                f"Insufficient confident samples: pos={len(pos_idx)}, neg={len(neg_idx)}. "
                f"Try adjusting thresholds or ensure predictions have good coverage."
            )
        
        # Combine
        sel_idx = np.concatenate([pos_idx, neg_idx])
        labels = np.concatenate([
            np.ones(len(pos_idx)),
            np.zeros(len(neg_idx)),
        ]).astype(np.float32)
        
        # Shuffle
        perm = rng.permutation(len(sel_idx))
        sel_idx = sel_idx[perm]
        labels = labels[perm]
        
        print(
            f"Final self-train set: N={len(sel_idx)} "
            f"(pos={int(labels.sum())}, neg={int((1-labels).sum())})"
        )
        
        return sel_idx, labels
    
    def prepare_selftrain_data(
        self,
        df: pd.DataFrame,
        model_path: str,
        window_size: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare self-training data from predictions CSV.
        
        Args:
            df: DataFrame with 'P', 'file', and probability columns
            model_path: Path to model (for finding probability column)
            window_size: Window size for creating windows
            
        Returns:
            Tuple of (X, y) arrays
        """
        if "P" not in df.columns:
            raise ValueError("DataFrame must include 'P' column")
        
        prob_col = self.find_probability_column(df, model_path)
        print(f"Using probability column: {prob_col}")
        
        # Collect windows and aligned probabilities
        X_all = []
        prob_centers_all = []
        
        if "file" in df.columns:
            groups = df.groupby("file")
        else:
            groups = [("all", df)]
        
        for file_id, group in groups:
            if "t_sec" in group.columns:
                group = group.sort_values("t_sec")
            
            power = group["P"].to_numpy(dtype=np.float32)
            prob = (
                group[prob_col].to_numpy(dtype=np.float32)
                if prob_col in group.columns
                else np.full(len(power), np.nan, dtype=np.float32)
            )
            
            if len(power) < window_size:
                continue
            
            # Create windows
            X = make_windows(power, window_size, normalize=True)[0]
            center = window_size // 2
            
            # Align probabilities with window centers
            prob_center = prob[center : center + len(X)]
            
            # Keep only windows with valid probabilities
            mask = ~np.isnan(prob_center)
            if mask.sum() == 0:
                continue
            
            X_all.append(X[mask])
            prob_centers_all.append(prob_center[mask])
        
        if not X_all:
            raise RuntimeError(
                "No windows produced. Check predictions CSV and probability column."
            )
        
        X_all = np.concatenate(X_all, axis=0)
        prob_centers = np.concatenate(prob_centers_all, axis=0)
        
        print(f"Total windows: {len(X_all)}")
        print(
            f"Prob stats: min={prob_centers.min():.3f}, "
            f"mean={prob_centers.mean():.3f}, "
            f"max={prob_centers.max():.3f}, "
            f"std={prob_centers.std():.3f}"
        )
        
        # Select confident samples
        sel_idx, labels = self.select_confident_samples(prob_centers)
        X_sel = X_all[sel_idx]
        
        return X_sel, labels
    
    def fine_tune(
        self,
        predictions_csv: str,
        model_path: str,
        output_path: str,
        window_size: int = 5,
    ):
        """
        Fine-tune model using self-training.
        
        Args:
            predictions_csv: Path to predictions CSV
            model_path: Path to pre-trained model
            output_path: Path to save fine-tuned model
            window_size: Window size
        """
        # Load predictions
        df = pd.read_csv(predictions_csv)
        
        # Prepare self-training data
        X, y = self.prepare_selftrain_data(df, model_path, window_size)
        
        # Create dataset from pre-windowed data
        dataset = NILMDataset.from_windows(X, y)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        # Create training config for fine-tuning
        train_config = TrainingConfig(
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            epochs=self.config.epochs,
            pos_weight_auto=True,
        )
        
        # Create trainer
        trainer = Trainer(self.model, train_config, device=self.device)
        
        # Load pre-trained weights
        trainer.load_model(model_path)
        
        # Fine-tune
        print("Starting fine-tuning...")
        history = trainer.train(loader, val_loader=None, save_dir=None)
        
        # Save fine-tuned model
        trainer.save_model(output_path)
        print(f"Fine-tuning complete. Model saved to: {output_path}")
