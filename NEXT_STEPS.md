# NILM Framework: Next Steps

This document consolidates all "what to do next" guidance from the original documentation.

---

## 1. Quick Model Improvements

_Source: `QUICK_IMPROVEMENTS.md`_

# Quick Model Improvements Guide

## TL;DR: How to Improve Your Model Right Now

Your current model has **F1 ~0.44** (best appliance). Here's how to get it to **0.60+** with minimal changes:

## 1. Update Configuration (5 minutes)

Edit `configs/default_config.yaml`:

```yaml
model:
  name: "ImprovedSeq2PointCNN"  # Use improved model
  window_size: 5
  input_channels: 1
  conv_channels: [32, 64, 128, 64]  # Deeper: was [16, 32]
  conv_kernel_size: 3
  conv_padding: 1
  hidden_dim: 64  # Larger: was 32
  output_dim: 1
  activation: "relu"
  dropout: 0.3  # Add regularization: was 0.0
  use_batch_norm: true  # Add batch norm

training:
  batch_size: 256
  learning_rate: 0.001
  epochs: 30  # Train longer: was 8
  seed: 42
  # Add learning rate scheduler
  scheduler: "ReduceLROnPlateau"
  scheduler_patience: 3
  scheduler_factor: 0.5
```

## 2. Use Improved Model (Already Created!)

The improved model is in `nilm_framework/models/seq2point_improved.py`. Just update your training script:

```python
# In scripts/train.py, change:
from nilm_framework.models import ImprovedSeq2PointCNN  # Instead of Seq2PointCNN

# Then use:
model = ImprovedSeq2PointCNN(
    window_size=config.model.window_size,
    conv_channels=config.model.conv_channels,
    use_batch_norm=True,
    dropout=config.model.dropout,
    hidden_dim=config.model.hidden_dim,
)
```

## 3. Expected Results

| Metric | Current | After Improvements |
|--------|---------|-------------------|
| Best F1 | 0.44 | **0.60-0.70** |
| Average F1 | 0.25 | **0.40-0.50** |
| Training Time | ~5 min/appliance | ~15 min/appliance |

## 4. Why These Changes Help

1. **Deeper Network** (`[32, 64, 128, 64]`):
   - More capacity to learn complex patterns
   - Better feature extraction

2. **Batch Normalization**:
   - Stabilizes training
   - Allows higher learning rates
   - Reduces internal covariate shift

3. **Dropout (0.3)**:
   - Prevents overfitting
   - Better generalization

4. **Longer Training (30 epochs)**:
   - Model needs more time to converge
   - 8 epochs is too short

5. **Learning Rate Scheduler**:
   - Automatically reduces LR when stuck
   - Better convergence

## 5. Test It

```bash
# Train with improved model
python scripts/train.py \
    --data Dataset/Exports/train_mix_5s.csv \
    --output outputs/training_improved \
    --config configs/default_config.yaml
```

Compare results:
- Old: `outputs/training/training_results.csv`
- New: `outputs/training_improved/training_results.csv`

## 6. Further Improvements (If Needed)

If F1 is still < 0.60 after these changes:

1. **Try larger windows**: `window_size: 10` (50 seconds)
2. **Use residual network**: `DeepSeq2PointCNN`
3. **Add data augmentation**: Noise, time shifts
4. **Ensemble models**: Train multiple and average predictions

## 7. Monitor Training

Watch for:
- ✅ Validation F1 increasing steadily
- ✅ Training loss decreasing smoothly
- ⚠️ Large gap between train/val = overfitting (increase dropout)
- ⚠️ Loss not decreasing = learning rate too high

## Summary

**Current Model**: Simple, fast, but limited (F1 ~0.44)
**Improved Model**: Deeper, regularized, longer training (F1 ~0.60-0.70)

The improvements are **already implemented** in the codebase. Just update your config and use `ImprovedSeq2PointCNN` instead of `Seq2PointCNN`!

---

## 2. Training Output Analysis & Action Plan

_Source: `TRAINING_ANALYSIS_AND_NEXT_STEPS.md`_

# Training Output Analysis & Next Steps

## 1. Training Results Summary

### Improved model (outputs/training_results.csv)

| Tier | Appliances | F1 range | Count |
|------|------------|----------|-------|
| **Strong** (F1 ≥ 0.70) | Impact_Drill, Microwave_On, AC_Adapter_Sony, Phone_Charger_Asus, Smoke_Extractor, LED_Panel | 0.67–0.92 | 6 |
| **Good** (0.40 ≤ F1 < 0.70) | Incandescent_Lamp, Oil_Heater_Q/R, Soldering_Station, Hair_Dryers, Phone_Charger_Motorola, etc. | 0.40–0.56 | 10 |
| **Weak** (F1 < 0.40) | LED_Lamp, Microwave_Standby, Resistor, Fan_Heater, Fan_3Speed, etc. | 0.12–0.39 | 7 |

### Top 10 by F1 (validation)

| Appliance | F1 | Precision | Recall | Pos rate |
|-----------|-----|-----------|--------|----------|
| y_Impact_Drill_P0 | **0.924** | 0.86 | 1.00 | 9.5% |
| y_Microwave_On_S0 | **0.843** | 0.73 | 1.00 | 5.0% |
| y_AC_Adapter_Sony_M0 | **0.773** | 0.73 | 0.83 | 19.6% |
| y_Phone_Charger_Asus_G0 | **0.737** | 0.60 | 0.95 | 3.1% |
| y_Smoke_Extractor_E0 | **0.685** | 0.54 | 0.93 | 13.6% |
| y_LED_Panel_D0 | **0.670** | 0.54 | 0.90 | 9.5% |
| y_Incandescent_Lamp_N0 | 0.557 | 0.41 | 0.87 | 20.0% |
| y_Oil_Heater_Q0 | 0.552 | 0.40 | 0.92 | 16.8% |
| y_Oil_Heater_R0 | 0.508 | 0.35 | 0.91 | 5.0% |
| y_Soldering_Station_H0 | 0.493 | 0.36 | 0.79 | 18.6% |

### Observations

- **Improvement over baseline:** The improved model clearly does better than the old 2-layer CNN (e.g. Incandescent_Lamp 0.44→0.56, AC_Adapter 0.43→0.77).
- **Recall bias:** Many appliances have recall ≈ 1.0 and precision lower. The model often over-predicts ON (more FPs), which is typical with class-weighted BCE.
- **Low–pos-rate appliances:** Fan_3Speed (2.3% pos), Phone_Charger_Asus (3.1%), etc. are harder; some still have good F1 (e.g. Phone_Charger_Asus 0.74).
- **23 appliances trained**; all have enough samples. Training itself looks healthy.

---

## 2. What We Have vs What We Need

| Item | Status |
|------|--------|
| **Trained models** (improved) | ✅ `outputs/models/` – 23 appliances |
| **Natural aggregate data** | ✅ `Exports/dataset_undersampled_5s/lit_natural_5s.csv` (file, t_sec, P) |
| **Predictions on natural** from **new** models | ❌ Not done yet |
| **Fine-tuning** of new models | ❌ Not done (and requires natural predictions first) |

Existing `Exports/.../natural_predictions.csv` and fine-tuned models under `Exports/` come from the **old** pipeline (older models). For the **new** improved models, we have not yet run inference on natural data.

---

## 3. Decision: What to Do Next

### Pipeline order

```
Train (synthetic/simulated) → Inference on natural → [Optional] Fine-tune on natural
```

Fine-tuning uses **predictions on natural data** as pseudo-labels. So we **must** run inference before we can fine-tune.

### Recommendation

1. **Do not retrain from scratch.**  
   Validation results are good; no need to re-train unless you change data or architecture.

2. **Next step: run inference on natural data**  
   Use the **new** models in `outputs/models/` and natural data `Exports/dataset_undersampled_5s/lit_natural_5s.csv`.  
   - Produces `outputs/natural_predictions.csv` (or similar).  
   - Lets you check how the improved model generalizes to real aggregate power.  
   - Produces the inputs needed for fine-tuning.

3. **After inference: decide on fine-tuning**  
   - **Strong appliances (F1 ≥ 0.70):** Inspect plots and histograms. If predictions look reasonable, fine-tuning can still help adapt to the natural distribution.  
   - **Good (F1 0.40–0.70):** Fine-tuning is typically useful (domain gap synthetic → natural).  
   - **Weak (F1 < 0.40):** Fine-tuning may help a bit, but gains are often limited; consider prioritising stronger appliances or future data/architecture improvements.

---

## 4. Concrete Next Steps

### Step 1: Run inference on natural data (do this first)

```bash
python scripts/inference.py \
  --data Exports/dataset_undersampled_5s/lit_natural_5s.csv \
  --model-dir outputs/models \
  --output outputs/natural_predictions.csv \
  --plot-dir outputs/plots \
  --config configs/default_config.yaml
```

- Use `--data` and `--model-dir` paths that match your project root (e.g. run from `Dataset` or project root, depending on where `Exports/` lives).
- This will:
  - Run all 23 improved models on natural aggregate power.
  - Save per-appliance probabilities in `outputs/natural_predictions.csv`.
  - Save histograms and overlay plots in `outputs/plots/` if `--plot-dir` is set.

### Step 2: Inspect outputs

- Open `outputs/natural_predictions.csv` and check that:
  - Probabilities are not all 0 or all 1.
  - Different appliances show different distributions.
- Look at `outputs/plots/` (e.g. histograms, overlay plots) to see if ON/OFF structure looks plausible.

### Step 3: Fine-tune selected appliances (optional, after Step 1)

For appliances you care about (e.g. top 5–10 by F1, or those you need for demos):

```bash
python scripts/finetune.py \
  --predictions outputs/natural_predictions.csv \
  --model outputs/models/cnn_seq2point_y_Incandescent_Lamp_N0.pt \
  --output outputs/models/cnn_seq2point_y_Incandescent_Lamp_N0_finetuned_natural.pt \
  --config configs/default_config.yaml
```

Repeat for other appliances (e.g. AC_Adapter_Sony, Soldering_Station, Oil_Heater_Q0, etc.) as needed.

### Step 4: (Optional) Re-run inference with fine-tuned models

Point `--model-dir` to a directory that contains both base and fine-tuned `.pt` files (with the same naming convention you use for fine-tuned models). The inference script will typically prefer fine-tuned weights when present. Then regenerate plots and compare with pre–fine-tuning results.

---

## 5. Summary

| Question | Answer |
|----------|--------|
| **Retrain from scratch?** | **No.** Training is successful; keep current runs. |
| **Run inference on natural data?** | **Yes.** Do this next with models in `outputs/models/`. |
| **Fine-tune?** | **After inference.** Use new predictions; fine-tune prioritised appliances. |

**Bottom line:** Run inference on natural data first. Use that output to decide which models to fine-tune and to sanity-check the improved model on real aggregate power.

---

## 3. Follow-up Ideas

In addition to the concrete steps above, see the "Implementation Priority" and "Next Steps" sections inside the design document for longer-term research directions (larger windows, residual networks, Transformers, data augmentation, and ensembles).
