"""Model architectures for NILM."""

from .seq2point import Seq2PointCNN
from .base import BaseNILMModel

# Improved models (optional import)
try:
    from .seq2point_improved import ImprovedSeq2PointCNN, DeepSeq2PointCNN
    __all__ = [
        "BaseNILMModel",
        "Seq2PointCNN",
        "ImprovedSeq2PointCNN",
        "DeepSeq2PointCNN",
    ]
except ImportError:
    __all__ = [
        "BaseNILMModel",
        "Seq2PointCNN",
    ]
