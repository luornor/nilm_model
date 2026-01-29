## 2. Training Output Analysis & Action Plan

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
| **Trained models** (improved) | ✅ `outputs/training/models/` – 23 appliances |
| **Natural aggregate data** | ✅ `Exports/dataset_undersampled_5s/lit_natural_5s.csv` (file, t_sec, P) |
| **Predictions on natural** from **new** models | ✅ `outputs/inference/natural_predictions.csv` + plots in `outputs/inference/plots/` |
| **Fine-tuning** of new models | ❌ Not done yet (requires natural predictions; ready now) |

Existing files under `Exports/` (older `natural_predictions.csv` and fine-tuned models) belong to the **old** pipeline and should be treated as legacy.

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

## 4. Concrete Next Steps (Updated)

### Step 1: Inference on natural data (DONE, command for re-runs)

Natural predictions for the improved models have already been generated to `outputs/inference/natural_predictions.csv` with plots in `outputs/inference/plots/` using:

```bash
py -3.11 scripts/inference.py \
  --data Exports/dataset_undersampled_5s/lit_natural_5s.csv \
  --model-dir outputs/training/models \
  --output outputs/inference/natural_predictions.csv \
  --plot-dir outputs/inference/plots \
  --config configs/default_config.yaml \
  --window-size 5
```

Re-use this command whenever you retrain models or after fine-tuning.

### Step 2: Appliance grouping based on natural probability distributions

Using `outputs/inference/natural_predictions.csv`, each appliance probability column was analyzed (mean, quantiles, and mass below 0.2 / above 0.8). Appliances are grouped as:

- **Almost always off** (very low mean, >97% of mass ≤ 0.2):
  - y_LED_Lamp_B0
  - y_Hair_Dryer_1900W_U0
  - y_Resistor_200ohm_L0
  - y_Microwave_Standby_A0
  - y_Fan_Heater_T0
  - y_Hair_Dryer_2100W_Z0

- **Mostly low** (dominant mass near 0, occasional spikes):
  - y_Hair_Dryer_2100W_Y0
  - y_Oil_Heater_R0
  - y_Fan_3Speed_K0
  - y_Hair_Dryer_1900W_V0
  - y_Hair_Dryer_2000W_X0
  - y_Hair_Dryer_2000W_W0
  - y_AC_Adapter_Sony_M0
  - y_Impact_Drill_P0

- **Ambiguous / broad** (substantial mass in mid-range 0.2–0.8):
  - y_Phone_Charger_Asus_G0
  - y_Microwave_On_S0
  - y_Oil_Heater_Q0
  - y_Phone_Charger_Motorola_I0

- **Bimodal (good separation)** (clear mass near both 0 and 1, limited mid-range):
  - y_Incandescent_Lamp_N0
  - y_Soldering_Station_H0
  - y_LED_Panel_D0

- **Mostly high** (often ON, strong peak near 1):
  - y_Smoke_Extractor_E0

The raw numbers and derived categories are codified in `configs/per_appliance_thresholds.yaml`.

### Step 3: Per-appliance ON/OFF threshold tuning

From the grouping above, per-appliance hysteresis thresholds (ON/OFF) have been chosen and stored in `configs/per_appliance_thresholds.yaml`:

- **Bimodal / mostly_high appliances** (confident ON/OFF separation):
  - y_Incandescent_Lamp_N0, y_Soldering_Station_H0, y_LED_Panel_D0, y_Smoke_Extractor_E0
  - Thresholds: `on = 0.8`, `off = 0.2` (conservative ON, aggressive OFF to reduce false positives).

- **Ambiguous appliances** (broad distributions):
  - y_Phone_Charger_Asus_G0, y_Microwave_On_S0, y_Oil_Heater_Q0, y_Phone_Charger_Motorola_I0
  - Thresholds: `on = 0.6`, `off = 0.4` (moderate hysteresis, close to 0.5 but stabilised).

- **Almost_always_off / mostly_low appliances**:
  - Remaining appliances listed above (LED_Lamp_B0, fans, hair dryers, resistor, AC_Adapter_Sony, Impact_Drill, Microwave_Standby, Fan_Heater_T0, etc.)
  - Thresholds: `on = 0.6`, `off = 0.4` (keeps rare, confident spikes while avoiding noise).

These thresholds currently affect analysis/interpretation (plots and any downstream ON/OFF conversion). The global defaults in `configs/default_config.yaml` can be left as-is; per-appliance overrides live in `configs/per_appliance_thresholds.yaml` for future integration into plotting or evaluation.

### Step 4: Fine-tune selected appliances (commands to run)

Use the improved models from `outputs/training/models` and the new natural predictions at `outputs/inference/natural_predictions.csv`. For each appliance, run:

- Incandescent Lamp:
  - `py -3.11 scripts/finetune.py --config configs/default_config.yaml --predictions outputs/inference/natural_predictions.csv --model outputs/training/models/cnn_seq2point_y_Incandescent_Lamp_N0.pt --output outputs/training/models/finetuned/cnn_seq2point_y_Incandescent_Lamp_N0_finetuned_natural.pt --window-size 5`

- AC Adapter (Sony):
  - `py -3.11 scripts/finetune.py --config configs/default_config.yaml --predictions outputs/inference/natural_predictions.csv --model outputs/training/models/cnn_seq2point_y_AC_Adapter_Sony_M0.pt --output outputs/training/models/finetuned/cnn_seq2point_y_AC_Adapter_Sony_M0_finetuned_natural.pt --window-size 5`

- Soldering Station:
  - `py -3.11 scripts/finetune.py --config configs/default_config.yaml --predictions outputs/inference/natural_predictions.csv --model outputs/training/models/cnn_seq2point_y_Soldering_Station_H0.pt --output outputs/training/models/finetuned/cnn_seq2point_y_Soldering_Station_H0_finetuned_natural.pt --window-size 5`

- Oil Heaters (Q and R):
  - `py -3.11 scripts/finetune.py --config configs/default_config.yaml --predictions outputs/inference/natural_predictions.csv --model outputs/training/models/cnn_seq2point_y_Oil_Heater_Q0.pt --output outputs/training/models/finetuned/cnn_seq2point_y_Oil_Heater_Q0_finetuned_natural.pt --window-size 5`
  - `py -3.11 scripts/finetune.py --config configs/default_config.yaml --predictions outputs/inference/natural_predictions.csv --model outputs/training/models/cnn_seq2point_y_Oil_Heater_R0.pt --output outputs/training/models/finetuned/cnn_seq2point_y_Oil_Heater_R0_finetuned_natural.pt --window-size 5`

- Phone Chargers:
  - Asus: `py -3.11 scripts/finetune.py --config configs/default_config.yaml --predictions outputs/inference/natural_predictions.csv --model outputs/training/models/cnn_seq2point_y_Phone_Charger_Asus_G0.pt --output outputs/training/models/finetuned/cnn_seq2point_y_Phone_Charger_Asus_G0_finetuned_natural.pt --window-size 5`
  - Motorola: `py -3.11 scripts/finetune.py --config configs/default_config.yaml --predictions outputs/inference/natural_predictions.csv --model outputs/training/models/cnn_seq2point_y_Phone_Charger_Motorola_I0.pt --output outputs/training/models/finetuned/cnn_seq2point_y_Phone_Charger_Motorola_I0_finetuned_natural.pt --window-size 5`

- Smoke Extractor and LED Panel:
  - Smoke_Extractor_E0: `py -3.11 scripts/finetune.py --config configs/default_config.yaml --predictions outputs/inference/natural_predictions.csv --model outputs/training/models/cnn_seq2point_y_Smoke_Extractor_E0.pt --output outputs/training/models/finetuned/cnn_seq2point_y_Smoke_Extractor_E0_finetuned_natural.pt --window-size 5`
  - LED_Panel_D0: `py -3.11 scripts/finetune.py --config configs/default_config.yaml --predictions outputs/inference/natural_predictions.csv --model outputs/training/models/cnn_seq2point_y_LED_Panel_D0.pt --output outputs/training/models/finetuned/cnn_seq2point_y_LED_Panel_D0_finetuned_natural.pt --window-size 5`

- Microwave On and Impact Drill (optional but strong performers):
  - Microwave_On_S0: `py -3.11 scripts/finetune.py --config configs/default_config.yaml --predictions outputs/inference/natural_predictions.csv --model outputs/training/models/cnn_seq2point_y_Microwave_On_S0.pt --output outputs/training/models/finetuned/cnn_seq2point_y_Microwave_On_S0_finetuned_natural.pt --window-size 5`
  - Impact_Drill_P0: `py -3.11 scripts/finetune.py --config configs/default_config.yaml --predictions outputs/inference/natural_predictions.csv --model outputs/training/models/cnn_seq2point_y_Impact_Drill_P0.pt --output outputs/training/models/finetuned/cnn_seq2point_y_Impact_Drill_P0_finetuned_natural.pt --window-size 5`

You can prioritise a subset (e.g. Incandescent_Lamp, AC_Adapter_Sony, Soldering_Station, Oil_Heater_Q/R, chargers, Smoke_Extractor, LED_Panel) depending on time.

### Step 5: Re-run inference with fine-tuned models (after Step 4)

After some fine-tuning runs complete, re-run inference so that `scripts/inference.py` automatically picks up any `*_finetuned_natural.pt` weights in `outputs/training/models`:

```bash
py -3.11 scripts/inference.py \
  --data Exports/dataset_undersampled_5s/lit_natural_5s.csv \
  --model-dir outputs/training/models/finetuned \
  --output outputs/inference/natural_predictions_finetuned.csv \
  --plot-dir outputs/inference/plots_finetuned \
  --config configs/default_config.yaml \
  --window-size 5
```

Then compare histograms and ON/OFF overlays between `outputs/inference/plots` and `outputs/inference/plots_finetuned` to quantify improvements on natural data.

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
