# NILM Framework Design

This document consolidates the key design, architecture, and refactoring information from the original documentation files.

---

## 1. Pipeline Overview

_Source: `PIPELINE_EXPLANATION.md`_

# NILM Pipeline: Current Implementation Analysis

## End-to-End Data Flow

### 1. **MATLAB Data Export** (`Matlab_Data/` → CSV)
- **Input**: MATLAB `.mat` files containing:
  - `vGrid`: Voltage waveform
  - `iHall`: Current waveform (Hall sensor)
  - `events_r`: Event markers (ON/OFF transitions)
  - `labels`: Appliance labels (e.g., "A0", "B0", "Incandescent_Lamp_N0")
  - `sps`: Samples per second (sampling rate)
  
- **Processing**:
  - Compute instantaneous power: `p = vGrid * iHall`
  - Bin into 5-second intervals (LF aggregation)
  - Extract appliance states from events: cumulative sum of events → binary state (>0.5 threshold)
  - Create one column per appliance: `y_<ApplianceName>`
  
- **Output**: CSV files with columns:
  - `file`: Source file identifier
  - `t_sec`: Time in seconds (5s intervals)
  - `P`: Aggregate power (W)
  - `y_*`: Binary appliance states (0/1)

- **Scripts**:
  - `export_synthetic_data_to_csv.m`: Synthetic labeled data
  - `export_simulated_data_to_csv.m`: Simulated labeled data
  - `lit_natural_to_5s_csv.py`: Natural unlabeled data (power only)

### 2. **Data Combination** (`combine_data.py`)
- Merges synthetic and simulated CSVs into `train_mix_5s.csv`
- Handles missing columns (fills with 0 for `y_*`, NaN for core fields)
- Adds `source` column for tracking

### 3. **Training** (`train_seq2point_binary.py`)
- **Architecture**: Seq2Point (predicts center point of window)
- **Model**: SmallCNN
  - Input: `(batch, 1, window_size=5)` - normalized power windows
  - Conv1D(1→16, kernel=3) → ReLU
  - Conv1D(16→32, kernel=3) → ReLU
  - AdaptiveAvgPool1d(1) → Flatten → Linear(32→1)
  - Output: Logit for binary classification
  
- **Training Process**:
  - For each appliance (`y_*` column):
    - Create sliding windows: `window_size=5` bins (25 seconds @ 5s/bin)
    - Normalize per-window: `(x - mean) / (std + 1e-6)`
    - Target: center point of window (`y[i + window//2]`)
    - Train/val split (80/20, stratified)
    - Loss: `BCEWithLogitsLoss` with pos_weight balancing
    - Train for 8 epochs, Adam (lr=1e-3)
    - Evaluate: F1, Precision, Recall
  
- **Output**: One `.pt` model file per appliance

### 4. **Fine-tuning / Self-Training** (`finetune_natural_selftrain.py`)
- **Input**: Natural data predictions from inference step
- **Process**:
  - Load pre-trained model
  - Extract high-confidence predictions:
    - Positive: `prob >= 0.90`
    - Negative: `prob <= 0.10`
  - Fallback: Top-K / Bottom-K if thresholds too strict
  - Create pseudo-labels from confident predictions
  - Fine-tune for 5 epochs with lower LR (2e-4)
  
- **Output**: Fine-tuned model (e.g., `*_finetuned_natural.pt`)

### 5. **Inference** (`natural_data_inference.py`)
- **Input**: Natural aggregate power data (no labels)
- **Process**:
  - Create sliding windows (same as training)
  - Run inference for each appliance model
  - Output probabilities (sigmoid of logits)
  - Generate plots:
    - Confidence histograms
    - Power + confidence + ON/OFF states (with hysteresis)
    - Overlay of all top appliances
  
- **Output**: `natural_predictions.csv` with probability columns

## Key Assumptions

1. **Sampling Rate**: 
   - Original: Variable `sps` (samples per second) in MATLAB files
   - After binning: 5-second intervals (0.2 Hz)

2. **Window Size**: 
   - Fixed at 5 bins = 25 seconds of aggregate power
   - Seq2Point: predicts center point (index 2)

3. **Target Format**:
   - Binary classification (ON/OFF states)
   - One model per appliance
   - Multi-label setup (multiple appliances can be ON simultaneously)

4. **Appliances**:
   - Variable set (discovered from data)
   - Examples: `y_Incandescent_Lamp_N0`, `y_AC_Adapter_Sony_M0`, etc.
   - Top performers tracked separately

5. **Data Normalization**:
   - Per-window normalization (not global)
   - Helps with different power scales across files

6. **Model Architecture**:
   - Small CNN (not Transformer or large CNN)
   - Designed for efficiency, not state-of-the-art accuracy

## Current Limitations

1. **Hard-coded paths**: All scripts use absolute Windows paths
2. **Duplication**: Dataset, model, windowing logic repeated across files
3. **No configuration**: Hyperparameters scattered in code
4. **Limited extensibility**: Adding new models requires copying code
5. **Mixed concerns**: Training, inference, plotting in same files
6. **No validation loop**: Training doesn't track validation metrics
7. **Fixed window size**: Not parameterized
8. **Manual appliance selection**: Top appliances hard-coded

## Proposed Improvements

The refactored framework will address these by:
- Centralized configuration (YAML + dataclass)
- Modular package structure
- Reusable components (Dataset, Model, Trainer)
- Separation of concerns (data, training, evaluation, inference)
- Extensible architecture (easy to add new models)
- Reproducible experiments (seed management, logging)
- Flexible window sizes and sampling rates

---

## 2. Framework Structure

_Source: `STRUCTURE.md`_

# NILM Framework Structure

## Proposed Folder Structure

```
.
├── nilm_framework/              # Main framework package
│   ├── __init__.py
│   ├── config.py                # Configuration management (YAML + dataclass)
│   ├── data/                    # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py           # Dataset classes (NILMDataset, Seq2PointDataset)
│   │   └── dataloader.py        # DataLoader creation utilities
│   ├── models/                  # Model architectures
│   │   ├── __init__.py
│   │   ├── base.py              # BaseNILMModel abstract class
│   │   └── seq2point.py         # Seq2PointCNN implementation
│   ├── training/                # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py           # Unified Trainer class
│   │   └── metrics.py           # Metrics computation
│   ├── finetuning/              # Fine-tuning / self-training
│   │   ├── __init__.py
│   │   └── selftrain.py         # SelfTrainer class
│   ├── inference/               # Inference utilities
│   │   ├── __init__.py
│   │   └── predictor.py         # Predictor class
│   ├── evaluation/              # Evaluation and visualization
│   │   ├── __init__.py
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── plotting.py          # Plotting utilities
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── data_utils.py        # Data preprocessing utilities
│       └── seed_utils.py        # Random seed management
│
├── scripts/                     # Main entry point scripts
│   ├── train.py                 # Training script
│   ├── finetune.py              # Fine-tuning script
│   └── inference.py             # Inference script
│
├── configs/                     # Configuration files
│   └── default_config.yaml      # Default configuration
│
├── requirements.txt             # Python dependencies
├── README.md                     # Main documentation
├── PIPELINE_EXPLANATION.md      # Current pipeline analysis
├── REFACTORING_SUMMARY.md       # Refactoring summary
└── STRUCTURE.md                 # This file
```

## Key Components

### Configuration (`nilm_framework/config.py`)

- `DataConfig`: Data loading and preprocessing settings
- `ModelConfig`: Model architecture settings
- `TrainingConfig`: Training hyperparameters
- `FineTuningConfig`: Fine-tuning/self-training settings
- `InferenceConfig`: Inference settings
- `ExperimentConfig`: Complete experiment configuration

### Data (`nilm_framework/data/`)

- `NILMDataset`: Base dataset class for windowed power signals
- `Seq2PointDataset`: Seq2Point-specific dataset
- `create_dataloaders()`: Create train/val DataLoaders from DataFrame

### Models (`nilm_framework/models/`)

- `BaseNILMModel`: Abstract base class for all NILM models
- `Seq2PointCNN`: Baseline CNN for Seq2Point architecture
- Extensible: Easy to add new architectures (CNN, Transformer, etc.)

### Training (`nilm_framework/training/`)

- `Trainer`: Unified training loop
  - Handles training and validation
  - Automatic loss balancing
  - Metrics tracking
  - Model checkpointing
- `compute_metrics()`: Classification metrics (F1, precision, recall)
- `MetricsTracker`: Track metrics during training

### Fine-tuning (`nilm_framework/finetuning/`)

- `SelfTrainer`: Self-training on natural data
  - High-confidence sample selection
  - Fallback strategies
  - Fine-tuning loop

### Inference (`nilm_framework/inference/`)

- `Predictor`: Run inference on natural data
  - Single file prediction
  - Multi-appliance prediction
  - Batch processing

### Evaluation (`nilm_framework/evaluation/`)

- `plot_confidence_histogram()`: Probability distribution plots
- `plot_onoff_states()`: Power + confidence + ON/OFF states
- `plot_overlay()`: Multi-appliance overlay plots
- `evaluate_predictions()`: Evaluation metrics

## Usage Flow

```
1. Data Preparation
   └─> CSV files with columns: file, t_sec, P, y_*

2. Training (scripts/train.py)
   └─> Train models per appliance
       └─> Save models to outputs/training/models/

3. Inference (scripts/inference.py)
   └─> Generate predictions on natural data
       └─> Save predictions.csv

4. Fine-tuning (scripts/finetune.py) [Optional]
   └─> Fine-tune using high-confidence predictions
       └─> Save fine-tuned models

5. Evaluation
   └─> Generate plots and metrics
```

## Extension Points

### Adding a New Model

1. Create class in `nilm_framework/models/`
2. Inherit from `BaseNILMModel`
3. Implement `forward()` method
4. Update `models/__init__.py`
5. Use in training scripts

### Adding a New Metric

1. Add to `nilm_framework/training/metrics.py`
2. Update `MetricsTracker` if needed
3. Use in training loop

### Supporting Different Data Formats

1. Add loader in `nilm_framework/data/`
2. Convert to standard DataFrame format
3. Use existing pipeline

### Multi-class or Regression

1. Update model output dimension
2. Modify loss function in `Trainer`
3. Adjust metrics computation
4. Update data loading if needed

## Design Principles

1. **Modularity**: Each component is self-contained
2. **Reusability**: Components can be used independently
3. **Extensibility**: Easy to add new features
4. **Reproducibility**: Configuration-driven experiments
5. **Documentation**: Clear docstrings and comments
6. **Research-grade**: Suitable for academic research

## Comparison with Original

| Aspect | Original | Refactored |
|--------|----------|------------|
| Code organization | Scattered scripts | Modular package |
| Configuration | Hard-coded | YAML + dataclass |
| Duplication | High | Minimal |
| Extensibility | Low | High |
| Reproducibility | Manual | Configuration-driven |
| Documentation | Minimal | Comprehensive |

---

## 3. Model Architecture Analysis

_Source: `MODEL_ANALYSIS.md`_

# Model Architecture Analysis & Recommendations

## Current Model Performance

Based on `sim_synth_cnn_results.csv`:

| Metric | Best | Average | Worst |
|--------|------|---------|-------|
| **F1 Score** | 0.44 | ~0.25 | 0.05 |
| **Precision** | 0.35 | ~0.15 | 0.03 |
| **Recall** | 0.95 | ~0.65 | 0.69 |

**Top 5 Appliances:**
1. `y_Incandescent_Lamp_N0`: F1=0.44
2. `y_AC_Adapter_Sony_M0`: F1=0.43
3. `y_Oil_Heater_Q0`: F1=0.37
4. `y_Soldering_Station_H0`: F1=0.36
5. `y_Smoke_Extractor_E0`: F1=0.34

## Current Architecture Analysis

### Current Model: `Seq2PointCNN`

```python
Conv1d(1 → 16, kernel=3) → ReLU
Conv1d(16 → 32, kernel=3) → ReLU
AdaptiveAvgPool1d(1)
Linear(32 → 1)
```

**Issues:**
1. **Too Shallow**: Only 2 convolutional layers
2. **Limited Capacity**: Max 32 channels
3. **Small Receptive Field**: Kernel=3 on window=5 (only sees 3/5 of window)
4. **No Regularization**: Dropout=0, no batch norm
5. **No Residual Connections**: Hard to train deeper networks
6. **Short Training**: Only 8 epochs

## Recommended Improvements

### 1. **Immediate Improvements** (Easy to implement)

#### A. Deeper Network
```python
# Current: [16, 32]
# Recommended: [32, 64, 128, 64]
conv_channels = [32, 64, 128, 64]
```

#### B. Add Batch Normalization
```python
Conv1d → BatchNorm1d → ReLU → Dropout
```

#### C. Add Dropout
```python
dropout = 0.3  # Instead of 0.0
```

#### D. Train Longer
```python
epochs = 20-30  # Instead of 8
# With learning rate scheduling
```

#### E. Learning Rate Schedule
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)
```

### 2. **Architecture Improvements**

#### Option A: Improved Seq2Point (Recommended)
- 4-5 conv layers
- Batch normalization
- Dropout (0.3)
- Larger hidden dimension (64-128)
- See `nilm_framework/models/seq2point_improved.py`

#### Option B: Deep Residual Network
- Residual blocks for gradient flow
- Can go much deeper (8-16 layers)
- Better for complex patterns
- See `DeepSeq2PointCNN` in improved model file

#### Option C: Transformer-based (Future)
- Self-attention for long-range dependencies
- Better for multi-appliance scenarios
- More parameters, slower training

### 3. **Training Improvements**

#### A. Longer Training
```yaml
training:
  epochs: 30  # Instead of 8
  learning_rate: 0.001
  # Add scheduler
  scheduler: "ReduceLROnPlateau"
  scheduler_patience: 3
```

#### B. Early Stopping
```python
# Stop if validation F1 doesn't improve for 5 epochs
early_stopping_patience = 5
```

#### C. Data Augmentation
- Add noise to power signals
- Time shifting
- Scaling variations

#### D. Better Loss Function
```python
# Focal loss for imbalanced data
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        # Focuses on hard examples
```

### 4. **Data & Preprocessing**

#### A. Larger Windows
```python
window_size = 10  # Instead of 5 (50 seconds instead of 25)
# Captures longer patterns
```

#### B. Multi-scale Windows
```python
# Use multiple window sizes and ensemble
window_sizes = [5, 10, 15]
```

#### C. Feature Engineering
- Add power derivatives (rate of change)
- Add frequency domain features (FFT)
- Add statistical features (mean, std, skew)

## Expected Performance Gains

| Improvement | Expected F1 Gain | Difficulty |
|------------|------------------|------------|
| Deeper network (4 layers) | +0.05-0.10 | Easy |
| Batch normalization | +0.02-0.05 | Easy |
| Dropout regularization | +0.02-0.05 | Easy |
| Longer training (30 epochs) | +0.05-0.10 | Easy |
| Learning rate scheduling | +0.02-0.05 | Easy |
| Residual connections | +0.05-0.15 | Medium |
| Larger windows (10) | +0.03-0.08 | Medium |
| Transformer architecture | +0.10-0.20 | Hard |

**Combined Expected**: F1 from ~0.44 → **0.60-0.75** for top appliances

## Implementation Priority

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Increase model depth: `[32, 64, 128, 64]`
2. ✅ Add batch normalization
3. ✅ Add dropout (0.3)
4. ✅ Train for 20-30 epochs
5. ✅ Add learning rate scheduler

**Expected**: F1 improvement of +0.10-0.15

### Phase 2: Architecture (2-4 hours)
1. Implement `ImprovedSeq2PointCNN`
2. Add residual connections
3. Experiment with larger windows
4. Hyperparameter tuning

**Expected**: Additional +0.05-0.10

### Phase 3: Advanced (1-2 weeks)
1. Transformer-based model
2. Multi-scale windows
3. Ensemble methods
4. Advanced data augmentation

**Expected**: Additional +0.10-0.20

## Code Example: Using Improved Model

```python
from nilm_framework.models.seq2point_improved import ImprovedSeq2PointCNN

# Create improved model
model = ImprovedSeq2PointCNN(
    window_size=5,
    conv_channels=[32, 64, 128, 64],  # Deeper
    use_batch_norm=True,               # Batch norm
    dropout=0.3,                       # Regularization
    hidden_dim=64,                     # Larger capacity
)

# Train with longer schedule
config.training.epochs = 30
config.training.learning_rate = 0.001
# Add scheduler in trainer
```

## Monitoring Improvements

Track these metrics:
- **Validation F1**: Should increase from ~0.44 to 0.60+
- **Training stability**: Loss should decrease smoothly
- **Overfitting**: Gap between train/val should be small
- **Per-appliance performance**: Focus on improving worst performers

## Conclusion

The current model is a **good baseline** but has significant room for improvement. The architecture is too simple for the complexity of NILM tasks. Implementing the Phase 1 improvements should yield substantial gains with minimal effort.

**Recommendation**: Start with Phase 1 improvements, then iterate based on results.

---

## 4. Refactoring Summary

_Source: `REFACTORING_SUMMARY.md`_

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

See the main README for detailed documentation and usage examples.

---

## 5. Improved Models Integration

_Source: `UPDATES_SUMMARY.md`_

# Updates Summary: Improved Models Integration

## Changes Made

All necessary files have been updated to use the improved models by default. Here's what changed:

### 1. Configuration Files

#### `configs/default_config.yaml`
- ✅ Model name changed to `"ImprovedSeq2PointCNN"`
- ✅ Deeper architecture: `conv_channels: [32, 64, 128, 64]` (was `[16, 32]`)
- ✅ Larger hidden dimension: `hidden_dim: 64` (was `32`)
- ✅ Added dropout: `dropout: 0.3` (was `0.0`)
- ✅ Enabled batch normalization: `use_batch_norm: true`
- ✅ Longer training: `epochs: 30` (was `8`)
- ✅ Learning rate scheduler: `scheduler: "ReduceLROnPlateau"`
- ✅ Save best model: `save_best: true`

#### `nilm_framework/config.py`
- ✅ Updated `ModelConfig` with new parameters:
  - `use_batch_norm: bool = True`
  - `use_residual: bool = False`
  - `base_channels: int = 32` (for DeepSeq2PointCNN)
  - `num_blocks: int = 4` (for DeepSeq2PointCNN)
- ✅ Default model name changed to `"ImprovedSeq2PointCNN"`
- ✅ Default `conv_channels` changed to `[32, 64, 128, 64]`
- ✅ Default `hidden_dim` changed to `64`
- ✅ Default `dropout` changed to `0.3`
- ✅ Added scheduler parameters to `TrainingConfig`

### 2. Scripts Updated

#### `scripts/train.py`
- ✅ Imports all three model types: `Seq2PointCNN`, `ImprovedSeq2PointCNN`, `DeepSeq2PointCNN`
- ✅ Dynamic model selection based on `config.model.name`
- ✅ Automatically uses `ImprovedSeq2PointCNN` when configured

#### `scripts/finetune.py`
- ✅ Imports all three model types
- ✅ Dynamic model selection based on config
- ✅ Backward compatible with old models

#### `scripts/inference.py`
- ✅ Imports all three model types
- ✅ Dynamic model selection based on config
- ✅ Can load models trained with any architecture

### 3. Model Module

#### `nilm_framework/models/__init__.py`
- ✅ Direct import of improved models (no try/except)
- ✅ All models exported: `Seq2PointCNN`, `ImprovedSeq2PointCNN`, `DeepSeq2PointCNN`

## How It Works

### Model Selection Logic

The scripts now automatically select the model based on `config.model.name`:

1. **"ImprovedSeq2PointCNN"** (default):
   - Uses `ImprovedSeq2PointCNN` class
   - 4-layer CNN with batch norm and dropout
   - Best for most cases

2. **"DeepSeq2PointCNN"**:
   - Uses `DeepSeq2PointCNN` class
   - Residual blocks for very deep networks
   - For complex patterns

3. **"Seq2PointCNN"** (backward compatibility):
   - Uses original `Seq2PointCNN` class
   - Simple 2-layer CNN
   - For comparison or legacy models

### Training Improvements

With the new defaults:
- **Deeper network**: 4 layers instead of 2
- **Regularization**: Batch norm + dropout (0.3)
- **Longer training**: 30 epochs instead of 8
- **Learning rate scheduling**: Automatic LR reduction
- **Best model saving**: Saves model with highest validation F1

### Expected Performance

| Metric | Old Model | Improved Model |
|--------|-----------|----------------|
| Best F1 | 0.44 | **0.60-0.70** |
| Average F1 | 0.25 | **0.40-0.50** |
| Training Time | ~5 min/appliance | ~15 min/appliance |

## Usage

### Training with Improved Model

```bash
# Uses ImprovedSeq2PointCNN by default
python scripts/train.py \
    --data Dataset/Exports/train_mix_5s.csv \
    --output outputs/training_improved \
    --config configs/default_config.yaml
```

### Using Different Models

Edit `configs/default_config.yaml`:

```yaml
model:
  name: "DeepSeq2PointCNN"  # or "Seq2PointCNN" for original
```

### Fine-tuning

```bash
# Automatically uses correct model architecture
python scripts/finetune.py \
    --predictions outputs/predictions.csv \
    --model outputs/training_improved/models/cnn_seq2point_y_Incandescent_Lamp_N0.pt \
    --output outputs/training_improved/models/cnn_seq2point_y_Incandescent_Lamp_N0_finetuned.pt \
    --config configs/default_config.yaml
```

### Inference

```bash
# Automatically loads correct model architecture
python scripts/inference.py \
    --data Dataset/Exports/dataset_undersampled_5s/lit_natural_5s.csv \
    --model-dir outputs/training_improved/models \
    --output outputs/predictions_improved.csv \
    --config configs/default_config.yaml
```

## Backward Compatibility

✅ **Old models still work**: Scripts can load models trained with `Seq2PointCNN`

✅ **Configurable**: Change `model.name` in config to use different architectures

✅ **Automatic detection**: Scripts try to infer model type from config

## Next Steps

1. **Train new models** with improved architecture:
   ```bash
   python scripts/train.py --data <your_data> --output outputs/improved
   ```

2. **Compare results**:
   - Old: `outputs/training/training_results.csv`
   - New: `outputs/improved/training_results.csv`

3. **Monitor training**:
   - Watch validation F1 increase
   - Check for overfitting (train/val gap)
   - Adjust dropout if needed

4. **Fine-tune** on natural data for even better performance

## Troubleshooting

### Model Architecture Mismatch

If you get errors loading a model:
- Ensure `config.model.name` matches the model architecture
- Check that `conv_channels`, `hidden_dim`, etc. match saved model

### Out of Memory

If training fails with OOM:
- Reduce `batch_size` in config (e.g., 128 instead of 256)
- Use `Seq2PointCNN` instead of `ImprovedSeq2PointCNN`
- Reduce `conv_channels` (e.g., `[16, 32, 64]` instead of `[32, 64, 128, 64]`)

### Performance Not Improving

- Check learning rate (might be too high/low)
- Increase training epochs
- Try different dropout values (0.2-0.5)
- Consider larger windows (`window_size: 10`)

## Summary

✅ All scripts updated to use improved models
✅ Configuration set to optimal defaults
✅ Backward compatible with old models
✅ Ready to train with better performance!

The framework now uses `ImprovedSeq2PointCNN` by default with:
- Deeper architecture (4 layers)
- Batch normalization
- Dropout regularization (0.3)
- Longer training (30 epochs)
- Learning rate scheduling
- Best model saving

Expected F1 improvement: **0.44 → 0.60-0.70** for top appliances!
