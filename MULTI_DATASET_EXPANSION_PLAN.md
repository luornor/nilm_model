# Multi-Dataset Expansion Plan for NILM Framework

## Executive Summary

Your current framework has:
- **23 trained appliances** from Matlab synthetic/simulated data (LIT dataset)
- **12 tested appliances** on PLAID natural data (plaid_eval_1s.csv)
- **Current performance**: Accuracy 59-78%, F1 scores 0.60-0.80

**Goal**: Expand to support 20+ additional device types from two new datasets while improving model generalization and performance.

---

## 1. Current State Analysis

### 1.1 Devices Currently Supported

**From plaid_eval_1s.csv (Natural data test results):**
- ✅ Hair_Iron (F1: 0.80, Acc: 78%)
- ✅ Laptop (F1: 0.68, Acc: 60%)
- ✅ Fridge (F1: 0.68, Acc: 64%)
- ✅ Vacuum (F1: 0.67, Acc: 65%)
- ✅ Fridge_defroster (F1: 0.67, Acc: 65%)
- ✅ Coffee_maker (F1: 0.65, Acc: 61%)
- ✅ Fan (F1: 0.65, Acc: 64%)
- ✅ Blender (F1: 0.65, Acc: 64%)
- ✅ Air_Conditioner (F1: 0.65, Acc: 59%)
- ✅ Incandescent_Light_Bulb (F1: 0.65, Acc: 63%)
- ✅ Compact_Fluorescent_Lamp (F1: 0.64, Acc: 60%)
- ✅ Water_kettle (F1: 0.64, Acc: 65%)

**Devices listed but not yet trained:**
- ❌ Iron
- ❌ Microwave
- ❌ Washing machine
- ❌ Airfryer
- ❌ Smart TV
- ❌ Deep freezer
- ❌ Speaker
- ❌ Electric stove
- ❌ Rice cooker
- ❌ Oven

### 1.2 New Dataset #1: Multi-Appliance Home Dataset (Image 1)

**Devices available:**
1. Air conditioner (1010W, Reactive) - **Already have similar**
2. Charger (120W, Switched-source) - **NEW TYPE**
3. Coffee maker (1000W, Resistive) - **Already have**
4. Computer (720W, Switched-source) - **NEW (different from laptop)**
5. Griddle (2200W, Resistive) - **NEW**
6. Hair dryer (2300W, Resistive) - **Similar to hair iron**
7. Heater (2000W, Resistive) - **NEW**
8. Iron (2800W, Resistive) - **NEW**
9. Laptop (360W, Switched-source) - **Already have**
10. Light (22W, Resistive) - **Already have**
11. Microwave (800W, Reactive) - **NEW**
12. Monitor (240W, Switched-source) - **NEW**
13. Vacuum (700W, Reactive) - **Already have**
14. Washing machine (2200W, Reactive/Resistive) - **NEW**

### 1.3 New Dataset #2: Room Occupancy Dataset (Image 2)

**Devices available:**
1. TVs (4 instances across rooms) - **NEW**
2. Laptops (2 instances) - **Already have**
3. PlayStation - **NEW (gaming console)**
4. Stove (725W) - **NEW**
5. Washing Machine (784W) - **Similar to dataset #1**
6. Oven (77W standby) - **NEW**
7. Refrigerator (484W) - **Already have**
8. Kettle (993W) - **Similar to water kettle**
9. Microwave (658W) - **Similar to dataset #1**
10. Coffee Machine (523W) - **Already have**
11. Freezer (84W) - **NEW (separate from fridge)**
12. Dishwasher (233W) - **NEW**
13. Water Heater (1223W) - **NEW**

---

## 2. Strategic Approach: Multi-Dataset Integration

### 2.1 Phase 1: Data Preparation & Harmonization (Weeks 1-2)

#### Step 1.1: Data Format Standardization

**Goal**: Convert both new datasets to your framework's CSV format.

**Required format:**
```csv
file,t_sec,P,y_<ApplianceName1>,y_<ApplianceName2>,...
dataset2_file1.mat,0,120.5,1,0,...
dataset2_file1.mat,1,125.3,1,0,...
```

**Action items:**
- [ ] **For Dataset #1** (Home appliances):
  - Identify the dataset source and format
  - Create `export_dataset1_to_csv.m` or `.py` script
  - Extract power signatures at 1-second resolution
  - Generate binary labels for each device
  - Save to `Exports/dataset1_1s.csv`

- [ ] **For Dataset #2** (Room occupancy):
  - This appears to be the CASAS or similar smart home dataset
  - Create `export_dataset2_to_csv.py` script
  - Parse room occupancy + appliance events
  - Convert to power + binary state format
  - Save to `Exports/dataset2_1s.csv`

**Implementation template:**
```python
# scripts/preprocess_dataset_new.py
import pandas as pd
import numpy as np

def parse_new_dataset(raw_data_path, output_csv):
    """
    Convert new dataset to NILM framework format.
    
    Required columns:
    - file: identifier
    - t_sec: time in seconds
    - P: aggregate power
    - y_*: binary appliance states
    """
    # TODO: Implement based on source format
    pass
```

#### Step 1.2: Device Name Mapping & Taxonomy

**Goal**: Create a unified device taxonomy across all datasets.

**Challenges:**
- Same device, different names (e.g., "Kettle" vs "Water_kettle")
- Same name, different power levels (e.g., Microwave 800W vs 658W)
- Device variants (e.g., "TV 1", "TV 2", "TV 3", "TV 4")

**Solution: Create device taxonomy file**

Create `configs/device_taxonomy.yaml`:
```yaml
# Device categories and mappings
device_taxonomy:
  heating:
    - Hair_Iron
    - Hair_Dryer_*
    - Heater
    - Griddle
    - Iron
    - Coffee_maker
    - Water_kettle
    - Kettle
    - Stove
  
  cooling:
    - Air_Conditioner
    - Fridge
    - Fridge_defroster
    - Freezer
    - Refrigerator
  
  entertainment:
    - Smart_TV
    - TV_*
    - Monitor
    - PlayStation
    - Speaker
    - Laptop
    - Computer
  
  kitchen:
    - Microwave
    - Oven
    - Coffee_maker
    - Coffee_Machine
    - Blender
    - Dishwasher
    - Rice_cooker
    - Washing_machine
    - Water_Heater
  
  motors:
    - Vacuum
    - Fan
    - Washing_machine
  
  lighting:
    - Incandescent_Light_Bulb
    - Compact_Fluorescent_Lamp
    - Light
  
  electronics:
    - Charger
    - Laptop
    - Computer
    - Monitor

# Device name mappings (normalize across datasets)
device_mappings:
  "Kettle": "Water_kettle"
  "Coffee Machine": "Coffee_maker"
  "Refrigerator": "Fridge"
  "TV 1": "TV_1"
  "TV 2": "TV_2"
  "TV 3": "TV_3"
  "TV 4": "TV_4"

# Power-based device differentiation
power_based_variants:
  Microwave:
    - name: "Microwave_800W"
      power_range: [700, 900]
      dataset: "dataset1"
    - name: "Microwave_650W"
      power_range: [550, 750]
      dataset: "dataset2"
  
  Washing_machine:
    - name: "Washing_machine_2200W"
      power_range: [2000, 2400]
      dataset: "dataset1"
    - name: "Washing_machine_780W"
      power_range: [600, 900]
      dataset: "dataset2"
```

#### Step 1.3: Data Quality Assessment

**Action items:**
- [ ] Analyze sampling rates (30kHz for PLAID data, check others)
- [ ] Check for missing values and outliers
- [ ] Verify event annotations quality
- [ ] Measure class imbalance (ON/OFF ratio) per device
- [ ] Calculate power distribution statistics

**Create assessment script:**
```python
# scripts/assess_data_quality.py
def assess_dataset_quality(csv_path):
    df = pd.read_csv(csv_path)
    
    report = {
        'total_samples': len(df),
        'duration_hours': df['t_sec'].max() / 3600,
        'appliances': {},
    }
    
    for col in df.columns:
        if col.startswith('y_'):
            appliance = col[2:]
            report['appliances'][appliance] = {
                'on_rate': df[col].mean(),
                'num_events': df[col].diff().abs().sum(),
                'avg_power_on': df[df[col] == 1]['P'].mean(),
                'avg_power_off': df[df[col] == 0]['P'].mean(),
            }
    
    return report
```

---

### 2.2 Phase 2: Model Architecture Improvements (Weeks 2-3)

#### Step 2.1: Multi-Scale Feature Extraction

**Problem**: Current model uses fixed 5-second window, which may not capture all device signatures.

**Solution**: Implement multi-scale CNN

```python
# nilm_framework/models/multiscale_seq2point.py

class MultiScaleSeq2Point(nn.Module):
    """
    Multi-scale feature extraction for diverse appliances.
    
    - Short-term (1s window): Fast-switching devices (lights, chargers)
    - Medium-term (5s window): Standard appliances (current default)
    - Long-term (15s window): Slow-varying devices (HVAC, refrigerator)
    """
    
    def __init__(self, window_sizes=[3, 5, 10], input_channels=1):
        super().__init__()
        
        self.branches = nn.ModuleList([
            self._create_branch(ws, input_channels) 
            for ws in window_sizes
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(sum([64 for _ in window_sizes]), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def _create_branch(self, window_size, in_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=min(3, window_size), padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=min(3, window_size), padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
    
    def forward(self, x_dict):
        """
        x_dict: {'window_3': tensor, 'window_5': tensor, 'window_10': tensor}
        """
        features = []
        for i, branch in enumerate(self.branches):
            x = x_dict[f'window_{[3, 5, 10][i]}']
            feat = branch(x).squeeze(-1)
            features.append(feat)
        
        combined = torch.cat(features, dim=1)
        return self.fusion(combined)
```

#### Step 2.2: Load-Type Aware Architecture

**Problem**: Different load types (Resistive, Reactive, Switched-source) have different power signatures.

**Solution**: Add load-type conditioning

```python
# nilm_framework/models/loadaware_seq2point.py

class LoadTypeAwareSeq2Point(nn.Module):
    """
    Conditions model on expected load type.
    
    Load types:
    - Resistive: Constant power (heaters, lights)
    - Reactive: Phase shift (motors, AC)
    - Switched-source: Pulsed (electronics, chargers)
    """
    
    def __init__(self, window_size=5, num_load_types=3):
        super().__init__()
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Load type embedding
        self.load_type_embedding = nn.Embedding(num_load_types, 32)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, load_type_idx):
        """
        x: (batch, 1, window_size)
        load_type_idx: (batch,) - 0: Resistive, 1: Reactive, 2: Switched
        """
        feat = self.features(x).squeeze(-1)
        load_emb = self.load_type_embedding(load_type_idx)
        
        combined = torch.cat([feat, load_emb], dim=1)
        return self.classifier(combined)
```

**Update config:**
```yaml
# configs/default_config.yaml (add new section)

model:
  name: "LoadTypeAwareSeq2Point"  # or "MultiScaleSeq2Point"
  load_type_aware: true
  multi_scale: false
  
# Load type mappings
load_types:
  resistive: 0
  reactive: 1
  switched_source: 2
  
appliance_load_types:
  y_Hair_Iron: resistive
  y_Air_Conditioner: reactive
  y_Laptop: switched_source
  y_Charger: switched_source
  y_Microwave: reactive
  y_Washing_machine: reactive
  # ... add all devices
```

---

### 2.3 Phase 3: Transfer Learning Strategy (Weeks 3-4)

#### Step 3.1: Cross-Dataset Transfer Learning

**Goal**: Leverage existing trained models to bootstrap learning on new datasets.

**Strategy:**

```
Source Domain (LIT dataset)          Target Domain (New datasets)
┌─────────────────────┐              ┌─────────────────────┐
│ 23 trained devices  │   Transfer   │ 20+ new devices     │
│ Synthetic/Simulated │─────────────>│ Real measurements   │
│ F1: 0.40-0.92       │   Learning   │ Different homes     │
└─────────────────────┘              └─────────────────────┘
```

**Approach A: Device-Specific Transfer**

For devices that exist in both datasets (e.g., Laptop, Vacuum, Coffee_maker):

```python
# scripts/transfer_learning.py

def transfer_to_new_device(source_model_path, target_data_csv, 
                           source_device, target_device):
    """
    Transfer learning from source to target device.
    
    Use when devices are similar (e.g., Laptop_LIT -> Laptop_Dataset2)
    """
    # Load source model
    source_model = load_model(source_model_path)
    
    # Freeze early layers (feature extraction)
    for param in source_model.features.parameters():
        param.requires_grad = False
    
    # Only fine-tune classifier
    optimizer = torch.optim.Adam(
        source_model.classifier.parameters(),
        lr=1e-4  # Lower LR
    )
    
    # Train on target data
    train_loader = create_dataloader(target_data_csv, target_device)
    
    for epoch in range(10):  # Fewer epochs
        train_one_epoch(source_model, train_loader, optimizer)
    
    return source_model
```

**Approach B: Category-Based Transfer**

For new devices in similar categories (e.g., Griddle -> use Heater model):

```python
def category_transfer(category, new_device_data):
    """
    Transfer from best model in category to new device.
    
    Example: 
    - Category: 'heating'
    - Best source: Hair_Iron (F1: 0.80)
    - Target: Griddle (NEW)
    """
    taxonomy = load_config('device_taxonomy.yaml')
    
    # Find best model in category
    category_devices = taxonomy['device_taxonomy'][category]
    best_model = select_best_model(category_devices)
    
    # Transfer and fine-tune
    return transfer_to_new_device(best_model, new_device_data, ...)
```

**Approach C: Multi-Task Learning**

Train a single model on all devices simultaneously:

```python
# nilm_framework/models/multitask_seq2point.py

class MultiTaskSeq2Point(nn.Module):
    """
    Single shared feature extractor + per-appliance heads.
    Learns general power signature features.
    """
    
    def __init__(self, window_size=5, num_appliances=40):
        super().__init__()
        
        # Shared feature extractor (transfer knowledge)
        self.shared_features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Per-appliance classifier heads
        self.appliance_heads = nn.ModuleDict({
            f'appliance_{i}': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            ) for i in range(num_appliances)
        })
    
    def forward(self, x, appliance_id):
        shared_feat = self.shared_features(x).squeeze(-1)
        return self.appliance_heads[f'appliance_{appliance_id}'](shared_feat)
```

#### Step 3.2: Domain Adaptation Techniques

**Problem**: Distribution shift between datasets (different homes, sampling rates, noise).

**Solution: Domain-Adversarial Training**

```python
# nilm_framework/training/domain_adaptation.py

class DomainAdaptiveSeq2Point(nn.Module):
    """
    Domain-adversarial neural network for NILM.
    
    Makes features invariant to dataset source.
    """
    
    def __init__(self, window_size=5):
        super().__init__()
        
        # Feature extractor (domain-invariant)
        self.feature_extractor = nn.Sequential(...)
        
        # Appliance classifier
        self.appliance_classifier = nn.Sequential(...)
        
        # Domain discriminator (adversarial)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Binary: LIT vs New dataset
        )
    
    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        
        # Appliance prediction
        appliance_pred = self.appliance_classifier(features)
        
        # Domain prediction (with gradient reversal)
        domain_pred = self.domain_discriminator(
            GradientReversalLayer.apply(features, alpha)
        )
        
        return appliance_pred, domain_pred

# Loss function
def domain_adaptive_loss(appliance_pred, appliance_true, 
                        domain_pred, domain_true, alpha=1.0):
    """
    alpha: balances appliance vs domain loss
    """
    appliance_loss = F.binary_cross_entropy_with_logits(
        appliance_pred, appliance_true
    )
    domain_loss = F.binary_cross_entropy_with_logits(
        domain_pred, domain_true
    )
    
    return appliance_loss + alpha * domain_loss
```

---

### 2.4 Phase 4: Data Augmentation for New Devices (Week 4)

#### Step 4.1: Synthetic Data Generation

**Goal**: Generate synthetic training data for devices with limited real samples.

**Technique 1: Power Profile Mixup**

```python
# nilm_framework/data/augmentation.py

def power_mixup(signal1, signal2, alpha=0.5):
    """
    Combine two power signals to create synthetic aggregate.
    
    Useful for creating multi-appliance scenarios.
    """
    return alpha * signal1 + (1 - alpha) * signal2

def appliance_profile_augmentation(df, appliance, num_augmented=1000):
    """
    Create variations of appliance signatures.
    
    - Add Gaussian noise
    - Scale power levels (±10%)
    - Time warping
    """
    augmented_samples = []
    
    on_segments = df[df[f'y_{appliance}'] == 1]
    
    for _ in range(num_augmented):
        sample = on_segments.sample(1)
        power = sample['P'].values[0]
        
        # Random augmentations
        noise = np.random.normal(0, power * 0.05)  # 5% noise
        scale = np.random.uniform(0.9, 1.1)  # ±10% scale
        
        aug_power = power * scale + noise
        augmented_samples.append(aug_power)
    
    return augmented_samples
```

**Technique 2: GAN-Based Signature Generation**

```python
# nilm_framework/models/signature_gan.py

class ApplianceSignatureGAN:
    """
    Generate realistic appliance power signatures.
    
    Useful when you have <100 real samples for a new device.
    """
    
    def __init__(self, signature_length=100):
        # Generator: noise -> power signature
        self.generator = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, signature_length),
            nn.Tanh(),
        )
        
        # Discriminator: signature -> real/fake
        self.discriminator = nn.Sequential(
            nn.Linear(signature_length, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
```

#### Step 4.2: Cross-Dataset Sample Balancing

**Problem**: Dataset #1 might have 10,000 samples for Microwave, Dataset #2 only 500.

**Solution: Balanced sampling strategy**

```python
# nilm_framework/data/balanced_sampler.py

class MultiDatasetBalancedSampler:
    """
    Sample equally from multiple datasets to prevent bias.
    """
    
    def __init__(self, datasets, samples_per_dataset=1000):
        self.datasets = datasets
        self.samples_per_dataset = samples_per_dataset
    
    def sample(self):
        batches = []
        for dataset in self.datasets:
            if len(dataset) < self.samples_per_dataset:
                # Oversample if needed
                indices = np.random.choice(
                    len(dataset), 
                    self.samples_per_dataset, 
                    replace=True
                )
            else:
                # Undersample
                indices = np.random.choice(
                    len(dataset), 
                    self.samples_per_dataset, 
                    replace=False
                )
            batches.append(dataset[indices])
        
        return np.concatenate(batches)
```

---

### 2.5 Phase 5: Evaluation & Benchmarking (Week 5)

#### Step 5.1: Cross-Dataset Evaluation Protocol

**Goal**: Fairly evaluate model performance across all datasets.

```python
# scripts/evaluate_cross_dataset.py

def cross_dataset_evaluation(model, datasets):
    """
    Comprehensive evaluation strategy:
    
    1. Within-dataset: Train on dataset A, test on dataset A
    2. Cross-dataset: Train on dataset A, test on dataset B
    3. Mixed: Train on A+B, test on C
    """
    
    results = {
        'within_dataset': {},
        'cross_dataset': {},
        'mixed_dataset': {},
    }
    
    # Within-dataset evaluation
    for ds_name, ds_data in datasets.items():
        train, test = train_test_split(ds_data)
        model.fit(train)
        metrics = model.evaluate(test)
        results['within_dataset'][ds_name] = metrics
    
    # Cross-dataset evaluation
    for train_ds in datasets:
        for test_ds in datasets:
            if train_ds != test_ds:
                model.fit(datasets[train_ds])
                metrics = model.evaluate(datasets[test_ds])
                results['cross_dataset'][f'{train_ds}->{test_ds}'] = metrics
    
    # Mixed dataset evaluation
    all_train = combine_datasets(datasets[:-1])
    model.fit(all_train)
    metrics = model.evaluate(datasets[-1])
    results['mixed_dataset'] = metrics
    
    return results
```

#### Step 5.2: Per-Device Performance Tracking

**Create performance dashboard:**

```python
# scripts/create_performance_dashboard.py

import plotly.graph_objects as go

def create_performance_dashboard(evaluation_results):
    """
    Interactive dashboard showing:
    - F1 scores per device across datasets
    - Confusion matrices
    - Power prediction error distributions
    - Cross-dataset transfer effectiveness
    """
    
    fig = go.Figure()
    
    # Heatmap: Devices vs Datasets (F1 scores)
    devices = list(evaluation_results.keys())
    datasets = ['LIT', 'PLAID', 'Dataset1', 'Dataset2']
    
    f1_matrix = np.array([
        [evaluation_results[dev][ds]['f1'] for ds in datasets]
        for dev in devices
    ])
    
    fig.add_trace(go.Heatmap(
        z=f1_matrix,
        x=datasets,
        y=devices,
        colorscale='RdYlGn',
    ))
    
    fig.write_html('outputs/performance_dashboard.html')
```

---

## 3. Implementation Roadmap

### Week 1-2: Data Integration
- [ ] Identify and acquire Dataset #1 (Home appliances)
- [ ] Identify and acquire Dataset #2 (Room occupancy)
- [ ] Create preprocessing scripts for both datasets
- [ ] Generate `Exports/dataset1_1s.csv` and `Exports/dataset2_1s.csv`
- [ ] Create `configs/device_taxonomy.yaml`
- [ ] Run data quality assessment
- [ ] Merge datasets into `Exports/combined_all_datasets.csv`

### Week 2-3: Model Architecture
- [ ] Implement `MultiScaleSeq2Point` model
- [ ] Implement `LoadTypeAwareSeq2Point` model
- [ ] Implement `MultiTaskSeq2Point` model
- [ ] Update `nilm_framework/models/` directory
- [ ] Create training scripts for new architectures
- [ ] Benchmark against current ImprovedSeq2PointCNN

### Week 3-4: Transfer Learning
- [ ] Implement device-specific transfer learning
- [ ] Implement category-based transfer
- [ ] Implement domain-adversarial training
- [ ] Create `scripts/transfer_learning.py`
- [ ] Test on devices present in multiple datasets
- [ ] Measure transfer effectiveness (target F1 vs from-scratch F1)

### Week 4: Data Augmentation
- [ ] Implement power mixup augmentation
- [ ] Implement signature noise injection
- [ ] (Optional) Implement GAN-based generation
- [ ] Create balanced sampling strategy
- [ ] Generate augmented training sets for rare devices

### Week 5: Evaluation & Iteration
- [ ] Run cross-dataset evaluation
- [ ] Create performance dashboard
- [ ] Identify underperforming devices
- [ ] Iterate on model architecture/hyperparameters
- [ ] Document best practices and lessons learned

---

## 4. Expected Outcomes

### 4.1 Device Coverage

**Current:**
- 12 devices tested on natural data
- 23 devices trained on synthetic data

**After expansion:**
- **40+ devices** supported across multiple categories
- **15+ new device types** added
- Coverage of all major home appliances

### 4.2 Performance Targets

| Metric | Current (PLAID 1s) | Target (After expansion) |
|--------|-------------------|-------------------------|
| Average F1 | 0.67 | **0.75** |
| Top-tier devices (F1 ≥ 0.80) | 1 (8%) | **10+ (25%)** |
| Weak devices (F1 < 0.50) | 0 (0%) | **<5 (12%)** |
| Cross-dataset generalization | N/A | **F1 > 0.65** |

### 4.3 Model Improvements

- **Multi-scale architecture**: Better capture diverse device signatures
- **Load-type awareness**: Exploit domain knowledge
- **Transfer learning**: Faster training for new devices
- **Domain adaptation**: Better generalization across homes
- **Data augmentation**: Handle rare device states

---

## 5. Risk Mitigation

### Risk 1: Dataset Incompatibility
**Mitigation**: 
- Start with data quality assessment before integration
- Use flexible preprocessing pipeline
- Create adapters for different formats

### Risk 2: Negative Transfer
**Problem**: Transfer learning may hurt performance if source/target too different.

**Mitigation**:
- Always benchmark transferred model vs from-scratch training
- Use selective fine-tuning (freeze/unfreeze layers)
- Implement early stopping based on validation loss

### Risk 3: Computational Resources
**Problem**: Training 40+ models is expensive.

**Mitigation**:
- Prioritize devices by importance/frequency
- Use multi-task learning (single model for all devices)
- Implement model distillation for deployment

### Risk 4: Overfitting to Dataset Quirks
**Problem**: Models learn dataset-specific artifacts.

**Mitigation**:
- Use k-fold cross-validation
- Test on held-out homes/buildings
- Apply domain randomization during training

---

## 6. Next Immediate Actions

### This Week:

1. **Identify Dataset Sources**
   - Search for the datasets from the images
   - Common candidates:
     - PLAID dataset (appliance-level)
     - BLUED dataset
     - UK-DALE dataset
     - REFIT dataset
     - CASAS smart home dataset (for room occupancy)
   
2. **Create Data Collection Plan**
   ```bash
   # Create dataset directories
   mkdir -p Dataset/Dataset1_Home_Appliances
   mkdir -p Dataset/Dataset2_Room_Occupancy
   
   # Documentation
   touch Dataset/Dataset1_Home_Appliances/README.md
   touch Dataset/Dataset2_Room_Occupancy/README.md
   ```

3. **Update Framework for Multi-Dataset Support**
   - Modify `config.py` to handle multiple data sources
   - Update `scripts/preprocess_data.py` to accept dataset parameter
   - Create unified data loader

4. **Baseline Testing**
   - Test current models on any new data samples you have
   - Measure cross-dataset performance drop
   - Identify biggest gaps

---

## 7. Long-Term Vision (3-6 months)

### 7.1 Deployment-Ready System

```
User inputs aggregate power → Model outputs all active devices
                 ↓
     [ Pre-trained on 40+ devices ]
                 ↓
     [ Fine-tuned on user's home ]
                 ↓
     [ Real-time inference <100ms ]
```

### 7.2 Continuous Learning Pipeline

- Collect user feedback on predictions
- Active learning: Request labels for uncertain predictions
- Incremental model updates
- A/B testing of model variants

### 7.3 Open-Source Contribution

- Package framework as pip-installable library
- Publish pre-trained models on Hugging Face
- Create documentation and tutorials
- Release benchmarks and leaderboards

---

## 8. Conclusion

This plan provides a **systematic path** to:
1. **Expand from 12 to 40+ devices**
2. **Improve average F1 from 0.67 to 0.75+**
3. **Enable cross-dataset generalization**
4. **Create a production-ready NILM system**

**Start with Phase 1** (data integration) and iterate. Each phase builds on the previous, allowing you to validate progress incrementally.

**Key success metric**: After implementing all phases, your model should achieve **F1 > 0.70** on unseen devices in unseen homes.
