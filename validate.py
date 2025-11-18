#!/usr/bin/env python3
"""
Quick validation script to ensure everything is working
"""
import sys
sys.path.insert(0, 'src')

from datasets.redd_loader import REDDLoader
from algorithms.combinatorial_optimization import CombinatorialOptimization

print("NILM VATA - Quick Validation")
print("="*50)

# Test 1: Data loading
print("\n[1/3] Testing data loader...")
loader = REDDLoader()
train_data, test_data = loader.get_train_test_split(train_ratio=0.7)
print(f"  ✓ Generated {len(train_data['mains'])} training samples")
print(f"  ✓ Generated {len(test_data['mains'])} testing samples")
print(f"  ✓ Loaded {len(train_data['appliances'])} appliances")

# Test 2: Training
print("\n[2/3] Testing CO algorithm training...")
co = CombinatorialOptimization(threshold=15.0)
co.train(train_data['mains'], train_data['appliances'])
print(f"  ✓ Trained models for {len(co.appliance_models)} appliances")

# Test 3: Disaggregation
print("\n[3/3] Testing disaggregation...")
predictions = co.disaggregate(test_data['mains'], num_samples=100)
print(f"  ✓ Disaggregated {len(predictions)} appliances")
print(f"  ✓ Each prediction has {len(predictions[list(predictions.keys())[0]])} samples")

print("\n" + "="*50)
print("✓ All tests passed!")
print("="*50)
print("\nYou can now run:")
print("  - python test_algorithms.py    (full testing)")
print("  - python demo_realtime.py      (interactive demo)")
print()
