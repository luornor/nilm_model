#!/usr/bin/env python
"""
Main training script for NILM models.

Trains one model per appliance on synthetic/simulated data.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nilm_framework.config import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig
from nilm_framework.models import Seq2PointCNN, ImprovedSeq2PointCNN, DeepSeq2PointCNN
from nilm_framework.data import create_dataloaders
from nilm_framework.training import Trainer
from nilm_framework.utils import find_appliance_columns, set_seed
from nilm_framework.training.metrics import compute_metrics
import torch


def train_appliance(
    df: pd.DataFrame,
    appliance: str,
    config: ExperimentConfig,
    output_dir: str,
) -> dict:
    """
    Train a single appliance model.
    
    Args:
        df: Training DataFrame
        appliance: Appliance column name (e.g., 'y_Incandescent_Lamp_N0')
        config: Experiment configuration
        output_dir: Output directory for models
        
    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*60}")
    print(f"Training model for: {appliance}")
    print(f"{'='*60}")
    
    # Filter valid data
    df_clean = df.dropna(subset=["P", appliance, "file"])
    
    if len(df_clean) == 0:
        print(f"  No valid data for {appliance}")
        return None
    
    # Check minimum requirements
    from nilm_framework.utils.data_utils import make_windows
    
    # Quick check on sample count
    sample_count = 0
    pos_count = 0
    for _, group in df_clean.groupby("file"):
        power = group["P"].to_numpy(dtype=np.float32)
        target = group[appliance].to_numpy(dtype=np.float32)
        if len(power) >= config.data.window_size:
            X, y = make_windows(
                power,
                config.data.window_size,
                target=target,
                normalize=config.data.normalize_per_window,
            )
            sample_count += len(X)
            if y is not None:
                pos_count += int(y.sum())
    
    if (sample_count < config.data.min_samples_per_appliance or
        pos_count < config.data.min_positives_per_appliance):
        print(
            f"  Skipping {appliance}: insufficient samples "
            f"(N={sample_count}, pos={pos_count})"
        )
        return None
    
    # Create data loaders
    try:
        train_loader, val_loader = create_dataloaders(
            df=df_clean,
            target_col=appliance,
            window_size=config.data.window_size,
            batch_size=config.training.batch_size,
            val_split=config.training.val_split,
            stratify=config.training.stratify,
            normalize=config.data.normalize_per_window,
            seed=config.training.seed,
        )
    except Exception as e:
        print(f"  Error creating data loaders: {e}")
        return None
    
    # Create model based on config
    model_name = config.model.name.lower()
    
    if model_name == "improvedseq2pointcnn":
        model = ImprovedSeq2PointCNN(
            window_size=config.model.window_size,
            input_channels=config.model.input_channels,
            conv_channels=config.model.conv_channels,
            conv_kernel_size=config.model.conv_kernel_size,
            use_batch_norm=config.model.use_batch_norm,
            dropout=config.model.dropout,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.output_dim,
            activation=config.model.activation,
            use_residual=config.model.use_residual,
        )
    elif model_name == "deepseq2pointcnn":
        model = DeepSeq2PointCNN(
            window_size=config.model.window_size,
            input_channels=config.model.input_channels,
            base_channels=config.model.base_channels,
            num_blocks=config.model.num_blocks,
            dropout=config.model.dropout,
            output_dim=config.model.output_dim,
        )
    else:  # Default to Seq2PointCNN
        model = Seq2PointCNN(
            window_size=config.model.window_size,
            input_channels=config.model.input_channels,
            conv_channels=config.model.conv_channels,
            conv_kernel_size=config.model.conv_kernel_size,
            conv_padding=config.model.conv_padding,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.output_dim,
            activation=config.model.activation,
            dropout=config.model.dropout,
        )
    
    # Create trainer
    trainer = Trainer(model, config.training)
    
    # Train
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    history = trainer.train(
        train_loader,
        val_loader=val_loader,
        save_dir=model_dir,
    )
    
    # Save model
    model_path = config.get_model_path(appliance, fine_tuned=False)
    trainer.save_model(model_path)
    
    # Get final validation metrics
    val_metrics = history["val_metrics"][-1] if history["val_metrics"] else {}
    
    result = {
        "appliance": appliance,
        "f1": val_metrics.get("f1", 0.0),
        "precision": val_metrics.get("precision", 0.0),
        "recall": val_metrics.get("recall", 0.0),
        "accuracy": val_metrics.get("accuracy", 0.0),
        "pos_rate": val_metrics.get("pos_rate", 0.0),
        "samples": sample_count,
        "positives": pos_count,
    }
    
    print(f"  Results: F1={result['f1']:.4f}, "
          f"Precision={result['precision']:.4f}, "
          f"Recall={result['recall']:.4f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Train NILM models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--appliances",
        type=str,
        nargs="+",
        default=None,
        help="Specific appliances to train (default: all)",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = ExperimentConfig.from_yaml(args.config)
    else:
        print(f"Config file not found: {args.config}, using defaults")
        config = ExperimentConfig()
        config.experiment_name = "nilm_training"
        config.output_dir = args.output
    
    # Override output dir
    config.output_dir = args.output
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set seed
    set_seed(config.training.seed)
    
    # Load data
    print(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Find appliances
    if args.appliances:
        appliances = args.appliances
    elif config.target_appliances:
        appliances = config.target_appliances
    else:
        appliances = find_appliance_columns(df)
    
    print(f"\nFound {len(appliances)} appliances to train")
    
    # Train each appliance
    results = []
    for appliance in appliances:
        result = train_appliance(df, appliance, config, config.output_dir)
        if result is not None:
            results.append(result)
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("f1", ascending=False)
        
        results_path = os.path.join(config.output_dir, "training_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Results saved to: {results_path}")
        print(f"{'='*60}")
        print("\nTop 10 appliances by F1 score:")
        print(results_df.head(10)[["appliance", "f1", "precision", "recall"]].to_string(index=False))
    else:
        print("\nNo models were trained successfully.")


if __name__ == "__main__":
    main()
