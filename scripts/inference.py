#!/usr/bin/env python
"""
Inference script for running predictions on natural data.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from glob import glob
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from nilm_framework.config import ExperimentConfig
from nilm_framework.models import Seq2PointCNN, ImprovedSeq2PointCNN, DeepSeq2PointCNN
from nilm_framework.inference import Predictor
from nilm_framework.evaluation import (
    plot_confidence_histogram,
    plot_onoff_states,
    plot_overlay,
)
from nilm_framework.utils import set_seed
import torch


def main():
    parser = argparse.ArgumentParser(description="Run inference on natural data")
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
        help="Path to natural data CSV",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save predictions CSV",
    )
    parser.add_argument(
        "--appliances",
        type=str,
        nargs="+",
        default=None,
        help="Specific appliances to predict (default: all in model_dir)",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="",
        help="Directory to save plots (optional)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Window size",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()
    
    set_seed(42)
    
    # Load data
    print(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows")
    
    # Find models
    if args.appliances:
        appliances = args.appliances
    else:
        # Auto-discover from model directory
        model_files = glob(os.path.join(args.model_dir, "cnn_seq2point_*.pt"))
        # Filter out fine-tuned models (prefer base models for inference)
        base_models = [f for f in model_files if "finetuned" not in f]
        appliances = []
        for model_file in base_models:
            # Extract appliance name from filename.
            # Expected format: "cnn_seq2point_y_ApplianceName.pt"
            basename = os.path.basename(model_file)
            prefix = "cnn_seq2point_"
            suffix = ".pt"
            if basename.startswith(prefix) and basename.endswith(suffix):
                # Strip the prefix and suffix to get the appliance name, e.g.
                # "cnn_seq2point_y_AC_Adapter_Sony_M0.pt" -> "y_AC_Adapter_Sony_M0"
                appliance = basename[len(prefix):-len(suffix)]
                appliances.append(appliance)
    
    print(f"Found {len(appliances)} appliances to predict")
    
    # Load models
    models = {}
    for appliance in appliances:
        # Try fine-tuned first, then base model
        fine_tuned_path = os.path.join(
            args.model_dir,
            f"cnn_seq2point_{appliance}_finetuned_natural.pt"
        )
        base_path = os.path.join(
            args.model_dir,
            f"cnn_seq2point_{appliance}.pt"
        )
        
        model_path = fine_tuned_path if os.path.exists(fine_tuned_path) else base_path
        
        if not os.path.exists(model_path):
            print(f"  Warning: Model not found for {appliance}")
            continue
        
        # Create model (architecture must match saved model)
        # Try to infer model type from config, default to ImprovedSeq2PointCNN
        model_name = config.model.name.lower() if hasattr(config.model, 'name') else "improvedseq2pointcnn"
        
        if model_name == "improvedseq2pointcnn":
            model = ImprovedSeq2PointCNN(
                window_size=args.window_size,
                input_channels=config.model.input_channels,
                conv_channels=config.model.conv_channels,
                conv_kernel_size=config.model.conv_kernel_size,
                use_batch_norm=getattr(config.model, 'use_batch_norm', True),
                dropout=getattr(config.model, 'dropout', 0.3),
                hidden_dim=config.model.hidden_dim,
                output_dim=config.model.output_dim,
                activation=config.model.activation,
            )
        elif model_name == "deepseq2pointcnn":
            model = DeepSeq2PointCNN(
                window_size=args.window_size,
                input_channels=config.model.input_channels,
                base_channels=getattr(config.model, 'base_channels', 32),
                num_blocks=getattr(config.model, 'num_blocks', 4),
                dropout=getattr(config.model, 'dropout', 0.3),
                output_dim=config.model.output_dim,
            )
        else:  # Default to Seq2PointCNN for backward compatibility
            model = Seq2PointCNN(
                window_size=args.window_size,
                input_channels=config.model.input_channels,
                conv_channels=config.model.conv_channels,
                conv_kernel_size=config.model.conv_kernel_size,
                conv_padding=config.model.conv_padding,
                hidden_dim=config.model.hidden_dim,
                output_dim=config.model.output_dim,
                activation=config.model.activation,
                dropout=getattr(config.model, 'dropout', 0.0),
            )
        
        # Load weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        models[appliance] = model
        print(f"  Loaded model for {appliance} from {os.path.basename(model_path)}")
    
    if not models:
        print("No models loaded. Exiting.")
        return
    
    # Run inference
    print("\nRunning inference...")
    pred_df = Predictor.predict_multiple_appliances(
        models=models,
        df=df,
        window_size=args.window_size,
        batch_size=256,
    )
    
    # Save predictions
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pred_df.to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output}")
    
    # Generate plots
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        print("\nGenerating plots...")
        
        # Get first file for plotting
        first_file = pred_df["file"].iloc[0] if "file" in pred_df.columns else None
        if first_file is not None:
            plot_df = pred_df[pred_df["file"] == first_file].copy()
            if "t_sec" in plot_df.columns:
                plot_df = plot_df.sort_values("t_sec")
            
            # Histograms
            for appliance in models.keys():
                prob_col = f"prob_{appliance}"
                if prob_col in plot_df.columns:
                    prob = plot_df[prob_col].to_numpy(dtype=np.float32)
                    plot_confidence_histogram(
                        prob,
                        title=f"Confidence Histogram ({appliance})",
                        output_path=os.path.join(args.plot_dir, f"hist_{appliance}.png"),
                    )
            
            # ON/OFF plots
            for appliance in models.keys():
                prob_col = f"prob_{appliance}"
                if prob_col in plot_df.columns:
                    plot_onoff_states(
                        plot_df,
                        appliance=appliance,
                        prob_col=prob_col,
                        on_threshold=config.inference.on_threshold,
                        off_threshold=config.inference.off_threshold,
                        smoothing_window=config.inference.smoothing_window,
                        output_path=os.path.join(args.plot_dir, f"onoff_{appliance}.png"),
                    )
            
            # Overlay plot
            plot_overlay(
                plot_df,
                appliances=list(models.keys()),
                output_path=os.path.join(args.plot_dir, "overlay_all.png"),
            )
        
        print(f"Plots saved to: {args.plot_dir}")


if __name__ == "__main__":
    main()
