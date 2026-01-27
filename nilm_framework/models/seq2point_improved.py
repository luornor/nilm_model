"""
Improved Seq2Point CNN model for NILM.

Enhanced version with:
- Deeper architecture (4-5 conv layers)
- Batch normalization
- Dropout for regularization
- Larger receptive field
- Residual connections (optional)
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .base import BaseNILMModel


class ImprovedSeq2PointCNN(BaseNILMModel):
    """
    Improved CNN for Seq2Point NILM.
    
    Architecture:
    - Multiple conv layers with batch norm
    - Dropout for regularization
    - Larger receptive field
    - More capacity for complex patterns
    """
    
    def __init__(
        self,
        window_size: int = 5,
        input_channels: int = 1,
        conv_channels: List[int] = None,
        conv_kernel_size: int = 3,
        use_batch_norm: bool = True,
        dropout: float = 0.3,
        hidden_dim: int = 64,
        output_dim: int = 1,
        activation: str = "relu",
        use_residual: bool = False,
    ):
        """
        Initialize Improved Seq2Point CNN.
        
        Args:
            window_size: Input sequence length
            input_channels: Number of input channels
            conv_channels: List of channel sizes (default: [32, 64, 128, 64])
            conv_kernel_size: Kernel size for conv layers
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout rate
            hidden_dim: Hidden dimension for final layers
            output_dim: Output dimension
            activation: Activation function
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        if conv_channels is None:
            conv_channels = [32, 64, 128, 64]  # Deeper than original [16, 32]
        
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
        
        for i, out_channels in enumerate(conv_channels):
            # Conv layer
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=conv_kernel_size,
                    padding=conv_kernel_size // 2,  # Same padding
                )
            )
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            
            # Activation
            layers.append(act_fn)
            
            # Dropout (except in first layer)
            if dropout > 0 and i > 0:
                layers.append(nn.Dropout(dropout))
            
            in_channels = out_channels
        
        # Adaptive pooling
        layers.append(nn.AdaptiveAvgPool1d(1))
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output head with more capacity
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels[-1], hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.use_residual = use_residual
    
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


class DeepSeq2PointCNN(BaseNILMModel):
    """
    Deep Seq2Point CNN with residual connections.
    
    Similar to ImprovedSeq2PointCNN but with residual blocks for
    better gradient flow and deeper networks.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        input_channels: int = 1,
        base_channels: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.3,
        output_dim: int = 1,
    ):
        """
        Initialize Deep Seq2Point CNN.
        
        Args:
            window_size: Input sequence length
            input_channels: Number of input channels
            base_channels: Base number of channels (doubles in each block)
            num_blocks: Number of residual blocks
            dropout: Dropout rate
            output_dim: Output dimension
        """
        super().__init__()
        
        # Initial conv
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        in_channels = base_channels
        
        for i in range(num_blocks):
            out_channels = base_channels * (2 ** min(i, 2))  # Cap at 4x
            self.blocks.append(
                ResidualBlock(in_channels, out_channels, dropout)
            )
            in_channels = out_channels
        
        # Final pooling and head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        return self.head(x)


class ResidualBlock(nn.Module):
    """Residual block for deep networks."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Projection for residual if channel size changes
        self.projection = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out
