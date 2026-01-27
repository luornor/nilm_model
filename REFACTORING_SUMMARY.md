# NILM Framework Refactoring Summary

## Overview

The NILM research pipeline has been refactored from a collection of standalone scripts into a clean, modular Python framework. This document summarizes the changes and improvements.

## Key Improvements

### 1. **Modular Package Structure**

**Before**: Scattered scripts with duplicated code
- `train_seq2point_binary.py`
- `finetune_natural_selftrain.py`
- `natural_data_inference.py`
- Each script duplicated Dataset, Model, and windowing logic

**After**: Organized package structure
```
nilm_framework/
├── config.py          # Centralized configuration
├── data/              # Data loading abstractions
├── models/            # Model architectures
├── training/          # Training utilities
├── finetuning/        # Fine-tuning/self-training
├── inference/         # Inference utilities
├── evaluation/        # Metrics and plotting
└── utils/             # Helper functions
```

### 2. **Configuration Management**

**Before**: Hard-coded paths and hyperparameters in each script
```python
CSV_PATH = r"C:\Users\ASUS\Desktop\Projects\ML Project\..."
WINDOW = 5
BATCH = 256
EPOCHS = 8
```

**After**: YAML-based configuration with dataclass defaults
```yaml
data:
  window_size: 5
  bin_size_seconds: 5.0
training:
  batch_size: 256
  epochs: 8
  learning_rate: 0.001
```

### 3. **Reusable Components**

**Before**: Duplicated code across scripts
- `Seq2PointDataset` defined 3 times
- `SmallCNN` defined 3 times
- `make_windows` logic duplicated

**After**: Single source of truth
- `NILMDataset` in `data/dataset.py`
- `Seq2PointCNN` in `models/seq2point.py`
- `make_windows` in `utils/data_utils.py`

### 4. **Separation of Concerns**

**Before**: Training, inference, and plotting mixed together

**After**: Clear separation
- `training/`: Training loop, metrics
- `inference/`: Prediction utilities
- `evaluation/`: Plotting and evaluation

### 5. **Extensibility**

**Before**: Adding new models required copying entire scripts

**After**: Easy to extend
- Inherit from `BaseNILMModel`
- Add to `models/__init__.py`
- Use existing training infrastructure

## Migration Guide

### Old Training Script

```python
# train_seq2point_binary.py
CSV_PATH = r"C:\Users\ASUS\Desktop\..."
WINDOW = 5
# ... hard-coded config ...

def train_one(df, target_col):
    # ... training logic ...
```

### New Training Script

```bash
python scripts/train.py \
    --data Dataset/Exports/train_mix_5s.csv \
    --output outputs/training \
    --config configs/default_config.yaml
```

Or programmatically:

```python
from nilm_framework.config import ExperimentConfig
from nilm_framework.models import Seq2PointCNN
from nilm_framework.data import create_dataloaders
from nilm_framework.training import Trainer

config = ExperimentConfig.from_yaml("configs/default_config.yaml")
train_loader, val_loader = create_dataloaders(df, target_col, ...)
model = Seq2PointCNN(...)
trainer = Trainer(model, config.training)
trainer.train(train_loader, val_loader)
```

## File Mapping

| Old File | New Location | Notes |
|---------|--------------|-------|
| `train_seq2point_binary.py` | `scripts/train.py` | Uses framework modules |
| `finetune_natural_selftrain.py` | `scripts/finetune.py` | Uses `finetuning.SelfTrainer` |
| `natural_data_inference.py` | `scripts/inference.py` | Uses `inference.Predictor` |
| Dataset class (duplicated) | `nilm_framework/data/dataset.py` | Single implementation |
| Model class (duplicated) | `nilm_framework/models/seq2point.py` | Extensible architecture |
| Window creation (duplicated) | `nilm_framework/utils/data_utils.py` | Reusable utility |

## Configuration Changes

### Old Approach
- Hard-coded in each script
- Windows-specific absolute paths
- No easy way to change hyperparameters

### New Approach
- YAML configuration files
- Relative paths (configurable)
- Easy to experiment with different settings

## Benefits

1. **Reproducibility**: Centralized configuration ensures experiments are reproducible
2. **Maintainability**: Single source of truth for each component
3. **Extensibility**: Easy to add new models, metrics, or features
4. **Testability**: Modular design enables unit testing
5. **Documentation**: Clear structure with docstrings
6. **Flexibility**: Support for different sampling rates, window sizes, etc.

## Backward Compatibility

The old scripts are preserved in `Dataset/Training/` for reference. The new framework:
- Uses the same data format (CSV with `file`, `t_sec`, `P`, `y_*` columns)
- Produces compatible model files (`.pt` PyTorch state dicts)
- Generates similar output CSVs and plots

## Next Steps

1. **Test the framework** with existing data
2. **Migrate experiments** to use new scripts
3. **Extend as needed**:
   - Add new model architectures
   - Support multi-class classification
   - Add regression support
   - Implement Transformer models

## Example Workflow

### 1. Train Models
```bash
python scripts/train.py \
    --data Dataset/Exports/train_mix_5s.csv \
    --output outputs/training
```

### 2. Run Inference
```bash
python scripts/inference.py \
    --data Dataset/Exports/dataset_undersampled_5s/lit_natural_5s.csv \
    --model-dir outputs/training/models \
    --output outputs/predictions.csv \
    --plot-dir outputs/plots
```

### 3. Fine-tune (Optional)
```bash
python scripts/finetune.py \
    --predictions outputs/predictions.csv \
    --model outputs/training/models/cnn_seq2point_y_Incandescent_Lamp_N0.pt \
    --output outputs/training/models/cnn_seq2point_y_Incandescent_Lamp_N0_finetuned.pt
```

## Questions?

See `README.md` for detailed documentation and usage examples.
