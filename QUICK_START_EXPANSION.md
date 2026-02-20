# Quick Start: Expanding Your NILM Framework

## What You Have Now

✅ **Main expansion plan**: [MULTI_DATASET_EXPANSION_PLAN.md](MULTI_DATASET_EXPANSION_PLAN.md)
✅ **Device taxonomy**: [configs/device_taxonomy.yaml](configs/device_taxonomy.yaml)
✅ **Preprocessing template**: [scripts/preprocess_new_dataset.py](scripts/preprocess_new_dataset.py)
✅ **Transfer learning analyzer**: [scripts/assess_transfer_learning_potential.py](scripts/assess_transfer_learning_potential.py)

## Immediate Action Plan (Next 7 Days)

### Step 1: Identify Your New Datasets (Day 1)

Based on the images you provided, you need to:

1. **Dataset #1 (Home Appliances)** - The table with 14 appliances
   - Looks like PLAID or similar dataset
   - Contains: Iron, Microwave, Washing machine, Computer, Monitor, etc.
   - **TODO**: Find the actual dataset source

2. **Dataset #2 (Room Occupancy)** - The room-based measurements
   - Looks like CASAS smart home dataset or similar
   - Contains: TVs, PlayStation, Stove, Dishwasher, Water Heater, etc.
   - **TODO**: Find the actual dataset source

**Action**: Search for these datasets:
```bash
# Common NILM datasets to check:
# - PLAID: https://energy.duke.edu/content/plaid
# - BLUED: http://portoalegre.andrew.cmu.edu:88/BLUED/
# - UK-DALE: https://jack-kelly.com/data/
# - REFIT: https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements
```

### Step 2: Download & Organize Data (Days 2-3)

Once you find the datasets:

```powershell
# Create dataset directories
mkdir Dataset\Dataset1_Home_Appliances
mkdir Dataset\Dataset2_Room_Occupancy

# Download data into these folders
# Then create README files documenting:
# - Dataset name and source
# - Number of devices
# - Sampling rate
# - File format
# - Any special notes
```

### Step 3: Preprocess First Dataset (Day 4)

Use the template I created:

```powershell
# Example: Preprocessing Dataset #1
py -3.11 scripts\preprocess_new_dataset.py `
  --dataset-path Dataset\Dataset1_Home_Appliances `
  --dataset-name dataset1 `
  --format custom `
  --output Exports\dataset1_1s.csv `
  --sampling-rate 1.0 `
  --validate `
  --stats
```

This will:
- Convert to standard format (file, t_sec, P, y_*)
- Normalize device names using taxonomy
- Validate data quality
- Generate statistics report

### Step 4: Assess Transfer Learning Potential (Day 5)

Compare your existing models with new devices:

```powershell
# Analyze which existing models can transfer to new devices
py -3.11 scripts\assess_transfer_learning_potential.py `
  --source-data Exports\lit_natural_5s_states.csv `
  --target-data Exports\dataset1_1s.csv `
  --output outputs\transfer_learning_assessment.json
```

This will tell you:
- Which existing models to use for each new device
- Transfer learning similarity scores
- Recommended training strategies

### Step 5: Start Training New Devices (Days 6-7)

**Option A: Train from scratch** (for devices with no good transfer candidate)

```powershell
py -3.11 scripts\train.py `
  --data Exports\dataset1_1s.csv `
  --config configs\default_config.yaml `
  --output outputs\training_dataset1
```

**Option B: Transfer learning** (for devices similar to existing ones)

```powershell
# TODO: Need to create scripts/transfer_learning.py
# For now, use fine-tuning approach:

# 1. Generate predictions on new data using closest existing model
py -3.11 scripts\inference.py `
  --data Exports\dataset1_1s.csv `
  --model-dir outputs\training\models `
  --output outputs\inference\dataset1_predictions.csv `
  --config configs\default_config.yaml

# 2. Fine-tune the model
py -3.11 scripts\finetune.py `
  --config configs\default_config.yaml `
  --predictions outputs\inference\dataset1_predictions.csv `
  --model outputs\training\models\cnn_seq2point_y_Laptop.pt `
  --output outputs\training\models\finetuned\cnn_seq2point_y_Computer_transferred.pt `
  --window-size 5
```

## Key Files Created

### 1. Device Taxonomy ([configs/device_taxonomy.yaml](configs/device_taxonomy.yaml))

Defines:
- **Device categories**: heating, cooling, entertainment, kitchen, motors, lighting, electronics
- **Name mappings**: Normalize "Kettle" → "Water_kettle", "TV 1" → "TV_1", etc.
- **Load types**: Resistive, Reactive, Switched-source
- **Transfer learning map**: Which existing models to use for new devices
- **Priority ranking**: Which devices to implement first

### 2. Preprocessing Template ([scripts/preprocess_new_dataset.py](scripts/preprocess_new_dataset.py))

Handles:
- Multiple dataset formats (CSV, MATLAB, HDF5, PLAID, BLUED, UK-DALE)
- Name normalization using taxonomy
- Data validation
- Statistics generation
- Quality assessment

### 3. Transfer Learning Analyzer ([scripts/assess_transfer_learning_potential.py](scripts/assess_transfer_learning_potential.py))

Computes:
- Power signature similarity between devices
- Best source models for each target device
- Transfer learning quality scores
- Automatic training command generation

## Expected Results

### Current Performance (plaid_eval_1s.csv)
- 12 devices tested
- Average F1: 0.67
- Best device: Hair_Iron (F1: 0.80)

### After Expansion (Target)
- **40+ devices** supported
- Average F1: **0.75+**
- **25%** of devices with F1 ≥ 0.80
- Cross-dataset generalization

## Architecture Improvements to Consider

### 1. Multi-Scale Architecture
Capture different time scales (1s, 5s, 15s windows)
- Better for devices with varying dynamics
- Implementation in MULTI_DATASET_EXPANSION_PLAN.md

### 2. Load-Type Aware Model
Condition model on device load type (Resistive/Reactive/Switched-source)
- Exploits domain knowledge
- Improves generalization

### 3. Multi-Task Learning
Single model for all devices with shared feature extractor
- Reduces training time
- Better feature learning
- Easier to deploy

## Troubleshooting

### Issue: Dataset format unknown
**Solution**: 
1. Check if it's a standard NILM dataset (PLAID, BLUED, etc.)
2. Look for metadata files (JSON, YAML, README)
3. Examine first few files manually
4. Update `preprocess_new_dataset.py` with custom parser

### Issue: Low transfer learning similarity (<0.3)
**Solution**: Train from scratch instead of transfer learning

### Issue: Poor performance on new dataset
**Possible causes**:
- Domain shift (different homes, different sampling rates)
- Missing normalization
- Insufficient training data

**Fixes**:
- Apply domain adaptation techniques
- Augment training data
- Fine-tune more aggressively

## Reference Documents

1. **[MULTI_DATASET_EXPANSION_PLAN.md](MULTI_DATASET_EXPANSION_PLAN.md)**
   - Complete 5-week roadmap
   - Detailed architecture proposals
   - Risk mitigation strategies

2. **[README.md](README.md)**
   - Current framework overview
   - Installation instructions
   - Basic usage

3. **[NEXT_STEPS.md](NEXT_STEPS.md)**
   - Current model performance
   - Fine-tuning recommendations
   - Improvement ideas

4. **[design.md](design.md)**
   - Pipeline architecture
   - Model details
   - Implementation notes

## Getting Help

If you encounter issues:
1. Check validation output from preprocessing
2. Review statistics JSON for data quality issues
3. Compare power signatures between datasets
4. Test on individual devices first before batch training

## Next Questions to Answer

1. **Which specific datasets are in your images?**
   - Need to identify exact source to download/access

2. **Do you have access to these datasets already?**
   - Or do we need to find alternative sources?

3. **What's your priority?**
   - More devices (coverage)?
   - Better accuracy (performance)?
   - Both equally?

4. **Computational constraints?**
   - How long can models train?
   - GPU available?

---

**Ready to start?** Begin with Step 1: Identify your datasets! 🚀
