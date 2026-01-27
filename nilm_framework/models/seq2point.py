"""
Seq2Point CNN model for NILM.

Predicts the appliance state at the center point of a sliding window.
"""

import torch
import torch.nn as nn
from typing import List
from .base import BaseNILMModel


class Seq2PointCNN(BaseNILMModel):
    """
    Small CNN for Seq2Point NILM.
    
    Architecture:
    - Conv1d(1 -> 16, kernel=3) -> ReLU
    - Conv1d(16 -> 32, kernel=3) -> ReLU
    - AdaptiveAvgPool1d(1)
    - Linear(32 -> 1)
    """
    
    def __init__(
        self,
        window_size: int = 5,
        input_channels: int = 1,
        conv_channels: List[int] = None,
        conv_kernel_size: int = 3,
        conv_padding: int = 1,
        hidden_dim: int = 32,
        output_dim: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """
        Initialize Seq2Point CNN.
        
        Args:
            window_size: Input sequence length (not used, kept for compatibility)
            input_channels: Number of input channels (typically 1 for power)
            conv_channels: List of channel sizes for conv layers
            conv_kernel_size: Kernel size for conv layers
            conv_padding: Padding for conv layers
            hidden_dim: Hidden dimension after pooling
            output_dim: Output dimension (1 for binary classification)
            activation: Activation function name
            dropout: Dropout rate (0.0 = no dropout)
        """
        super().__init__()
        
        if conv_channels is None:
            conv_channels = [16, 32]
        
        # Activation function
        if activation.lower() == "relu":
            act_fn = nn.ReLU()
        elif activation.lower() == "gelu":
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build convolutional layers
        layers = []
        in_channels = input_channels
        
        for out_channels in conv_channels:
            layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=conv_kernel_size,
                    padding=conv_padding,
                ),
                act_fn,
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        # Adaptive pooling to fixed size
        layers.append(nn.AdaptiveAvgPool1d(1))
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels[-1], output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, sequence_length)
            
        Returns:
            Output tensor (batch, output_dim)
        """
        features = self.feature_extractor(x)
        output = self.head(features)
        return output
