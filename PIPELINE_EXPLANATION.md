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
