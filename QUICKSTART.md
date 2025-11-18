# Quick Start Guide

Get started with NILM VATA in 5 minutes!

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Run Your First Test

```bash
python test_algorithms.py
```

This will:
- Generate synthetic energy data (simulating a home with fridge, microwave, and light)
- Train two NILM algorithms (CO and FHMM)
- Disaggregate the energy consumption
- Show you the results and accuracy metrics
- Generate visualization plots

**Expected output:**
```
Testing NILM Algorithms
======================================================================

[1/4] Loading dataset...
  Training samples: 60,480
  Testing samples: 25,920
  Appliances: fridge, microwave, light

[2/4] Visualizing power distributions...

[3/4] Testing Combinatorial Optimization...
Training Combinatorial Optimization model...
  fridge: ON=200.0W (±10.0), OFF=0.0W
  microwave: ON=1500.0W (±50.0), OFF=0.0W
  light: ON=60.0W (±5.0), OFF=0.0W

Disaggregating 10,000 test samples...

Combinatorial Optimization Results:
================================================================================
Appliance       MAE (W)      RMSE (W)     NDE        F1 Score
================================================================================
fridge          XX.XX        XX.XX        X.XXX      X.XXX
microwave       XX.XX        XX.XX        X.XXX      X.XXX
light           XX.XX        XX.XX        X.XXX      X.XXX
================================================================================
AVERAGE         XX.XX        XX.XX        X.XXX      X.XXX
================================================================================

...
```

## 3. View the Results

After running `test_algorithms.py`, you'll have three new image files:

1. **power_distributions.png** - Shows how much power each appliance uses
2. **results_co.png** - Shows how well the CO algorithm detected each appliance
3. **results_fhmm.png** - Shows how well the FHMM algorithm detected each appliance

Open these files to see the disaggregation results!

## 4. Try the Real-Time Demo

```bash
python demo_realtime.py
```

Then follow the interactive prompts:

```
Select NILM Algorithm:
  1. Combinatorial Optimization (CO)
  2. Factorial HMM (FHMM)
  3. Both (compare)

Enter choice (1-3) [default=1]: 1

Select demo type:
  1. Real-time simulation (live plotting)
  2. Batch disaggregation (full test set)

Enter choice (1-2) [default=1]: 1
```

You'll see a live plot showing how the algorithm disaggregates energy in real-time!

## 5. Understanding the Results

### What do the metrics mean?

- **MAE (Mean Absolute Error)**: Lower is better. Shows average error in watts.
- **RMSE (Root Mean Squared Error)**: Lower is better. Penalizes large errors more.
- **F1 Score**: Higher is better (0-1). Shows how well it detects ON/OFF states.
- **Energy Accuracy**: Higher is better (0-100%). Shows total energy estimation accuracy.

### What appliances are simulated?

The synthetic dataset includes:
- **Fridge**: Always running with ON/OFF cycles (~200W when on)
- **Microwave**: Occasional high-power bursts (~1500W)
- **Light**: ON during morning and evening (~60W)

### Which algorithm is better?

- **CO (Combinatorial Optimization)**: Faster, better for appliances with clear ON/OFF states
- **FHMM (Factorial HMM)**: Slower, better for complex patterns and temporal dependencies

Try both and compare!

## Next Steps

### Customize the Data

Edit `src/datasets/redd_loader.py` to:
- Add more appliances
- Change power consumption patterns
- Adjust data duration

### Use Real Data

To use actual REDD data instead of synthetic:
1. Download REDD from http://redd.csail.mit.edu/
2. Place in `./data/REDD/`
3. Modify `redd_loader.py` to read from files

### Tune Algorithm Parameters

**For CO:**
```python
co = CombinatorialOptimization(threshold=15.0)  # Adjust threshold
```

**For FHMM:**
```python
fhmm = FHMM(n_states=3)  # Try more states (2, 3, 4)
```

### Add Your Own Algorithm

1. Create `src/algorithms/my_algorithm.py`
2. Implement `train()` and `disaggregate()` methods
3. Add to demo scripts

See the README for detailed examples!

## Troubleshooting

### "No module named 'src'"

Make sure you're running from the project root directory:
```bash
cd nilm-vata
python test_algorithms.py
```

### Plots not showing

If running on a server without display:
```bash
# Set matplotlib to use non-interactive backend
export MPLBACKEND=Agg
python test_algorithms.py
```

### Slow execution

Reduce the number of samples:
```python
# In test_algorithms.py, change:
num_test_samples = min(1000, len(test_data['mains']))  # Instead of 10000
```

## Questions?

See the full README.md for detailed documentation!
