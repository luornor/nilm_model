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
