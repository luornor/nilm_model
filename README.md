# NILM Framework

A modular, extensible Python framework for Non-Intrusive Load Monitoring (NILM) research.

## Overview

This framework provides a clean, research-grade implementation for training and evaluating NILM models on energy disaggregation tasks. It supports:

- **Seq2Point architecture** (baseline CNN)
- **Binary classification** (ON/OFF state detection)
- **Multi-appliance** training and inference
- **Self-training / fine-tuning** on natural data
- **Extensible design** for adding new models and features

## Project Structure

```
nilm_framework/
├── __init__.py
├── config.py              # Configuration management
├── data/                  # Data loading and preprocessing
│   ├── dataset.py
│   └── dataloader.py
├── models/                # Model architectures
│   ├── base.py
│   └── seq2point.py
├── training/              # Training utilities
│   ├── trainer.py
│   └── metrics.py
├── finetuning/            # Fine-tuning / self-training
│   └── selftrain.py
├── inference/             # Inference utilities
│   └── predictor.py
├── evaluation/            # Evaluation and plotting
│   ├── metrics.py
│   └── plotting.py
└── utils/                 # Utility functions
    ├── data_utils.py
    └── seed_utils.py

scripts/
├── train.py               # Main training script
├── finetune.py            # Fine-tuning script
└── inference.py           # Inference script

configs/
└── default_config.yaml    # Default configuration
```

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy, Pandas
- scikit-learn
- matplotlib
- PyYAML

### Setup

```bash
# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib pyyaml

# Or if using conda
conda install pytorch numpy pandas scikit-learn matplotlib pyyaml -c pytorch
```

## Quick Start

### 1. Prepare Data

Ensure your data is in CSV format with columns:
- `file`: File identifier
- `t_sec`: Time in seconds
- `P`: Aggregate power (W)
- `y_*`: Binary appliance states (0/1)

Example:
```csv
file,t_sec,P,y_Incandescent_Lamp_N0,y_AC_Adapter_Sony_M0
file1.mat,0,120.5,1,0
file1.mat,5,125.3,1,0
...
```

### 2. Train Models

Train models on synthetic/simulated data:

```bash
python scripts/train.py \
    --data Dataset/Exports/train_mix_5s.csv \
    --output outputs/training \
    --config configs/default_config.yaml
```

This will:
- Train one model per appliance
- Save models to `outputs/training/models/`
- Generate `training_results.csv` with metrics

### 3. Run Inference

Generate predictions on natural data:

```bash
python scripts/inference.py \
    --data Dataset/Exports/dataset_undersampled_5s/lit_natural_5s.csv \
    --model-dir outputs/training/models \
    --output outputs/predictions.csv \
    --plot-dir outputs/plots \
    --window-size 5
```

### 4. Fine-tune (Optional)

Fine-tune models using self-training on natural data:

```bash
python scripts/finetune.py \
    --predictions outputs/predictions.csv \
    --model outputs/training/models/cnn_seq2point_y_Incandescent_Lamp_N0.pt \
    --output outputs/training/models/cnn_seq2point_y_Incandescent_Lamp_N0_finetuned_natural.pt \
    --window-size 5
```

## Configuration

The framework uses YAML configuration files. Key settings:

### Data Configuration
- `window_size`: Number of bins in sliding window (default: 5)
- `bin_size_seconds`: Time bin size (default: 5.0 seconds)
- `normalize_per_window`: Normalize each window independently

### Model Configuration
- `conv_channels`: CNN channel sizes (default: [16, 32])
- `hidden_dim`: Hidden dimension after pooling (default: 32)

### Training Configuration
- `batch_size`: Batch size (default: 256)
- `learning_rate`: Learning rate (default: 1e-3)
- `epochs`: Number of epochs (default: 8)
- `val_split`: Validation split ratio (default: 0.2)

### Fine-tuning Configuration
- `pos_threshold`: High-confidence positive threshold (default: 0.90)
- `neg_threshold`: High-confidence negative threshold (default: 0.10)
- `max_pos_samples`: Maximum positive samples (default: 6000)
- `max_neg_samples`: Maximum negative samples (default: 6000)

See `configs/default_config.yaml` for all options.

## Usage Examples

### Programmatic Usage

```python
from nilm_framework.config import ExperimentConfig
from nilm_framework.models import Seq2PointCNN
from nilm_framework.data import create_dataloaders
from nilm_framework.training import Trainer
from nilm_framework.utils import set_seed
import pandas as pd

# Load configuration
config = ExperimentConfig.from_yaml("configs/default_config.yaml")
set_seed(config.training.seed)

# Load data
df = pd.read_csv("data/train.csv")

# Create data loaders
train_loader, val_loader = create_dataloaders(
    df=df,
    target_col="y_Incandescent_Lamp_N0",
    window_size=config.data.window_size,
    batch_size=config.training.batch_size,
)

# Create model
model = Seq2PointCNN(
    window_size=config.model.window_size,
    conv_channels=config.model.conv_channels,
)

# Train
trainer = Trainer(model, config.training)
history = trainer.train(train_loader, val_loader=val_loader)
```

### Custom Models

To add a new model architecture:

1. Create a new model class inheriting from `BaseNILMModel`:

```python
from nilm_framework.models.base import BaseNILMModel
import torch.nn as nn

class MyCustomModel(BaseNILMModel):
    def __init__(self, window_size=5):
        super().__init__()
        # Define your architecture
        
    def forward(self, x):
        # Forward pass
        return output
```

2. Register it in `nilm_framework/models/__init__.py`

3. Use it in training scripts

## Data Format

### Training Data

CSV with columns:
- `file`: Source file identifier
- `t_sec`: Time in seconds (5s intervals)
- `P`: Aggregate power (W)
- `y_<ApplianceName>`: Binary appliance states (0/1)

### Natural Data (for inference)

CSV with columns:
- `file`: Source file identifier
- `t_sec`: Time in seconds
- `P`: Aggregate power (W)

No `y_*` columns required (unlabeled).

## Extending the Framework

### Adding New Models

1. Create model class in `nilm_framework/models/`
2. Inherit from `BaseNILMModel`
3. Implement `forward()` method
4. Update configuration if needed

### Adding New Metrics

1. Add metric computation in `nilm_framework/training/metrics.py`
2. Update `MetricsTracker` if needed
3. Use in training loop

### Supporting Different Sampling Rates

1. Update `bin_size_seconds` in configuration
2. Adjust `window_size` accordingly
3. Ensure data preprocessing handles the rate correctly

### Multi-class or Regression

1. Modify model output dimension
2. Update loss function in `Trainer`
3. Adjust metrics computation
4. Update data loading if needed

## Pipeline Overview

1. **Data Export** (MATLAB → CSV)
   - Export MATLAB `.mat` files to CSV format
   - Bin power signals into 5-second intervals
   - Extract appliance states from events

2. **Training**
   - Train one binary classifier per appliance
   - Use Seq2Point architecture (predicts center of window)
   - Evaluate on validation set

3. **Inference**
   - Run predictions on natural (unlabeled) data
   - Generate probability outputs
   - Create visualizations

4. **Fine-tuning** (Optional)
   - Use high-confidence predictions as pseudo-labels
   - Fine-tune pre-trained models
   - Improve performance on natural data

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in configuration
- Use smaller `window_size`
- Process files in smaller batches

### Poor Performance
- Check data quality and labeling
- Adjust hyperparameters (learning rate, epochs)
- Try different model architectures
- Increase training data

### Missing Models
- Ensure model paths are correct
- Check that training completed successfully
- Verify appliance names match between training and inference

## License

This framework is provided for research purposes.

## Citation

If you use this framework in your research, please cite appropriately.

## Contributing

Contributions are welcome! Please:
1. Follow the existing code structure
2. Add docstrings and comments
3. Update documentation
4. Test your changes
