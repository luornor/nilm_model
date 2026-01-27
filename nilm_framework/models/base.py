"""Base model class for NILM models."""

import torch.nn as nn
from abc import ABC, abstractmethod


class BaseNILMModel(nn.Module, ABC):
    """
    Base class for NILM models.
    
    All NILM models should inherit from this class.
    """
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, sequence_length)
            
        Returns:
            Output tensor (batch, output_dim)
        """
        pass
    
    def get_output_dim(self) -> int:
        """Get output dimension of the model."""
        # Try to infer from last layer
        for module in reversed(list(self.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features
        raise ValueError("Could not determine output dimension")
