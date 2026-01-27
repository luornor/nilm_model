"""Metrics computation for NILM evaluation."""

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
from typing import Dict, Tuple, Optional


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels (or probabilities if y_prob is None)
        y_prob: Predicted probabilities (optional)
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metrics
    """
    # Convert probabilities to predictions if needed
    if y_prob is not None:
        y_pred_binary = (y_prob >= threshold).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)
        y_prob = y_pred
    
    y_true = y_true.astype(int)
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred_binary)),
        "f1": float(f1_score(y_true, y_pred_binary, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_binary, zero_division=0)),
        "pos_rate": float(y_true.mean()),
        "pred_pos_rate": float(y_pred_binary.mean()),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        metrics["tn"] = int(tn)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)
        metrics["tp"] = int(tp)
    
    return metrics


class MetricsTracker:
    """Track metrics during training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.losses = []
        self.metrics_history = []
    
    def update(self, loss: float, metrics: Optional[Dict[str, float]] = None):
        """
        Update tracked metrics.
        
        Args:
            loss: Loss value
            metrics: Optional dictionary of metrics
        """
        self.losses.append(loss)
        if metrics:
            self.metrics_history.append(metrics)
    
    def get_mean_loss(self) -> float:
        """Get mean loss over tracked values."""
        return float(np.mean(self.losses)) if self.losses else 0.0
    
    def get_latest_metrics(self) -> Optional[Dict[str, float]]:
        """Get most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
