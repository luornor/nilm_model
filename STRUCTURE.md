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
