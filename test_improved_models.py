#!/usr/bin/env python
"""Quick test to verify improved models are properly integrated."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing improved models integration...")
print("=" * 60)

# Test 1: Import models
try:
    from nilm_framework.models import ImprovedSeq2PointCNN, DeepSeq2PointCNN, Seq2PointCNN
    print("✓ All models imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load configuration
try:
    from nilm_framework.config import ExperimentConfig
    config = ExperimentConfig.from_yaml('configs/default_config.yaml')
    print(f"✓ Config loaded: model={config.model.name}")
    print(f"  - dropout={config.model.dropout}")
    print(f"  - use_batch_norm={config.model.use_batch_norm}")
    print(f"  - conv_channels={config.model.conv_channels}")
    print(f"  - epochs={config.training.epochs}")
    print(f"  - scheduler={config.training.scheduler}")
except Exception as e:
    print(f"✗ Config loading failed: {e}")
    sys.exit(1)

# Test 3: Create improved model
try:
    model = ImprovedSeq2PointCNN(
        window_size=5,
        conv_channels=[32, 64, 128, 64],
        use_batch_norm=True,
        dropout=0.3,
        hidden_dim=64,
    )
    print("✓ ImprovedSeq2PointCNN created successfully")
    
    # Test forward pass
    import torch
    x = torch.randn(2, 1, 5)  # batch=2, channels=1, length=5
    out = model(x)
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {out.shape}")
    assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"
    print("  - Forward pass works correctly")
except Exception as e:
    print(f"✗ Model creation/forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create deep model
try:
    model = DeepSeq2PointCNN(
        window_size=5,
        base_channels=32,
        num_blocks=4,
        dropout=0.3,
    )
    print("✓ DeepSeq2PointCNN created successfully")
    
    # Test forward pass
    import torch
    x = torch.randn(2, 1, 5)
    out = model(x)
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {out.shape}")
    assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"
    print("  - Forward pass works correctly")
except Exception as e:
    print(f"✗ Deep model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("✓ All tests passed! Improved models are ready to use.")
print("\nNext steps:")
print("1. Train with: python scripts/train.py --data <your_data> --output outputs/improved")
print("2. Compare results with old models")
print("3. Fine-tune on natural data for even better performance")
