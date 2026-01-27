#!/usr/bin/env python
"""
Fine-tuning script using self-training on natural data.
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nilm_framework.config import ExperimentConfig, FineTuningConfig
from nilm_framework.models import Seq2PointCNN
from nilm_framework.finetuning import SelfTrainer
from nilm_framework.utils import set_seed
import torch


def main():
    parser = argparse.ArgumentParser(description="Fine-tune NILM models using self-training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions CSV from inference",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save fine-tuned model",
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
    
    # Create model (architecture must match pre-trained model)
    model = Seq2PointCNN(
        window_size=args.window_size,
        input_channels=config.model.input_channels,
        conv_channels=config.model.conv_channels,
        conv_kernel_size=config.model.conv_kernel_size,
        conv_padding=config.model.conv_padding,
        hidden_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim,
    )
    
    # Create self-trainer
    selftrainer = SelfTrainer(model, config.fine_tuning)
    
    # Fine-tune
    selftrainer.fine_tune(
        predictions_csv=args.predictions,
        model_path=args.model,
        output_path=args.output,
        window_size=args.window_size,
    )
    
    print(f"\nFine-tuning complete! Model saved to: {args.output}")


if __name__ == "__main__":
    main()
