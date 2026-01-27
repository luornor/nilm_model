"""Evaluation metrics for NILM predictions."""

import numpy as np
from typing import Dict, Optional
from ..training.metrics import compute_metrics


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate predictions against ground truth.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    return compute_metrics(y_true, y_prob, y_prob=y_prob, threshold=threshold)
