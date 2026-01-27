"""
Centralized configuration management for NILM framework.

Supports YAML configuration files and dataclass-based defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    # Paths
    dataset_root: str = "Dataset"
    matlab_data_dir: str = "Matlab_Data"
    exports_dir: str = "Exports"
    
    # Data characteristics
    bin_size_seconds: float = 5.0  # Time bin size for aggregation
    window_size: int = 5  # Number of bins in sliding window
    center_index: Optional[int] = None  # Auto-computed as window_size // 2
    
    # Normalization
    normalize_per_window: bool = True  # Normalize each window independently
    
    # Filtering
    min_samples_per_appliance: int = 200
    min_positives_per_appliance: int = 20
    
    def __post_init__(self):
        if self.center_index is None:
            self.center_index = self.window_size // 2


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    name: str = "Seq2PointCNN"
    window_size: int = 5  # Should match DataConfig.window_size
    
    # CNN architecture
    input_channels: int = 1
    conv_channels: List[int] = field(default_factory=lambda: [16, 32])
    conv_kernel_size: int = 3
    conv_padding: int = 1
    hidden_dim: int = 32
    output_dim: int = 1  # Binary classification
    
    # Activation
    activation: str = "relu"
    
    # Dropout (for future use)
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Hyperparameters
    batch_size: int = 256
    learning_rate: float = 1e-3
    epochs: int = 8
    seed: int = 42
    
    # Loss
    loss_type: str = "BCEWithLogitsLoss"
    pos_weight_auto: bool = True  # Auto-balance based on class distribution
    
    # Optimizer
    optimizer: str = "Adam"
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Learning rate scheduler
    scheduler: Optional[str] = None  # "ReduceLROnPlateau", "StepLR", or None
    scheduler_patience: int = 3  # For ReduceLROnPlateau
    scheduler_factor: float = 0.5  # For ReduceLROnPlateau
    scheduler_step_size: int = 10  # For StepLR
    scheduler_gamma: float = 0.1  # For StepLR
    
    # Validation
    val_split: float = 0.2
    stratify: bool = True
    
    # Logging
    log_interval: int = 100
    save_best: bool = False  # Save best model based on validation metric
    
    # Device
    device: Optional[str] = None  # None = auto-detect (cuda/cpu)


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning/self-training."""
    # Paths
    predictions_csv: str = ""  # Path to predictions from inference
    
    # Self-training thresholds
    pos_threshold: float = 0.90
    neg_threshold: float = 0.10
    max_pos_samples: int = 6000
    max_neg_samples: int = 6000
    min_confident_samples: int = 200
    
    # Training
    epochs: int = 5
    learning_rate: float = 2e-4
    batch_size: int = 256
    
    # Fallback strategy
    use_fallback_selection: bool = True  # Use top-K if thresholds too strict


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    # Paths
    natural_data_csv: str = ""
    model_dir: str = ""
    output_csv: str = ""
    
    # Appliances to predict
    target_appliances: List[str] = field(default_factory=list)  # Empty = all available
    
    # Post-processing
    smoothing_window: int = 5  # Moving average for probabilities
    on_threshold: float = 0.55
    off_threshold: float = 0.45
    
    # Plotting
    plot_histograms: bool = True
    plot_onoff_states: bool = True
    plot_overlay: bool = True
    plot_output_dir: str = ""


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str = "nilm_experiment"
    output_dir: str = "outputs"
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Appliances
    target_appliances: Optional[List[str]] = None  # None = auto-discover
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        # Handle nested configs
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        fine_tuning_config = FineTuningConfig(**config_dict.get("fine_tuning", {}))
        inference_config = InferenceConfig(**config_dict.get("inference", {}))
        
        return cls(
            experiment_name=config_dict.get("experiment_name", "nilm_experiment"),
            output_dir=config_dict.get("output_dir", "outputs"),
            data=data_config,
            model=model_config,
            training=training_config,
            fine_tuning=fine_tuning_config,
            inference=inference_config,
            target_appliances=config_dict.get("target_appliances")
        )
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            "experiment_name": self.experiment_name,
            "output_dir": self.output_dir,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "fine_tuning": self.fine_tuning.__dict__,
            "inference": self.inference.__dict__,
        }
        if self.target_appliances is not None:
            config_dict["target_appliances"] = self.target_appliances
        
        os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def get_model_path(self, appliance: str, fine_tuned: bool = False) -> str:
        """Get path to model file for an appliance."""
        suffix = "_finetuned_natural.pt" if fine_tuned else ".pt"
        filename = f"cnn_seq2point_{appliance}{suffix}"
        return os.path.join(self.output_dir, "models", filename)
