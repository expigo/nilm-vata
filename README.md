# NILM VATA - NILM Algorithm Testing and Visualization

A comprehensive framework for testing and comparing Non-Intrusive Load Monitoring (NILM) algorithms with real-time energy disaggregation capabilities.

## Overview

This project implements and compares multiple NILM algorithms for energy disaggregation:

- **Combinatorial Optimization (CO)**: Fast, deterministic approach using state combinations
- **Factorial Hidden Markov Model (FHMM)**: Probabilistic approach using HMMs

The framework includes:
- Synthetic REDD-like dataset generation
- Multiple NILM algorithms
- Real-time disaggregation simulation
- Comprehensive performance metrics
- Visualization tools
- Model persistence and management
- Energy cost calculation
- Streaming data processing

## Features

### Core Features
- ✅ Multiple NILM algorithms (CO, FHMM)
- ✅ REDD-like dataset support (synthetic data generation)
- ✅ Real-time disaggregation simulation
- ✅ Performance metrics (MAE, RMSE, F1, NDE)
- ✅ Interactive visualizations
- ✅ Batch and streaming processing modes
- ✅ Easy-to-use API

### Advanced Features
- ✅ **Model Persistence**: Save and load trained models
- ✅ **Energy Cost Calculator**: Calculate costs with time-of-use pricing
- ✅ **Benchmark Tool**: Compare algorithms with detailed reports
- ✅ **Streaming Support**: Real-time async processing
- ✅ **Cost Recommendations**: AI-powered energy saving suggestions

## Installation

### Requirements

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer

### Setup with uv (Recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd nilm-vata

# Install dependencies
uv pip install -e .

# Or use make for convenience
make install
```

### Alternative: Setup with pip

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Quick Start

### 1. Test Algorithms (Simple)

Run the basic test script to evaluate both algorithms:

```bash
python test_algorithms.py
```

This will:
- Generate synthetic REDD-like data
- Train CO and FHMM algorithms
- Evaluate performance on test data
- Generate visualization plots
- Compare algorithm performance

**Output files:**
- `power_distributions.png` - Appliance power histograms
- `results_co.png` - CO disaggregation results
- `results_fhmm.png` - FHMM disaggregation results

### 2. Real-Time Demo (Interactive)

Run the interactive real-time simulation:

```bash
python demo_realtime.py
# Or with make
make demo
```

This provides an interactive menu to:
1. Choose NILM algorithm (CO, FHMM, or both)
2. Select demo type (real-time or batch)
3. View live disaggregation with animated plots

### 3. Benchmark Algorithms (NEW!)

Compare multiple algorithms with comprehensive metrics:

```bash
python benchmark.py
# Or with make
make benchmark
```

This will:
- Train and test multiple algorithms
- Generate performance comparison charts
- Calculate energy costs
- Save trained models
- Provide recommendations

**Output files:**
- `benchmark_report.txt` - Detailed comparison report
- `benchmark_comparison.png` - Visual comparison charts
- `energy_cost_breakdown.png` - Cost analysis
- `models/*.pkl` - Saved trained models

### 4. Quick Validation

```bash
python validate.py
# Or with make
make validate
```

## Using Make Commands

The project includes a Makefile for convenience:

```bash
make help          # Show all available commands
make install       # Install dependencies with uv
make test          # Run test algorithms
make demo          # Run real-time demo
make benchmark     # Run benchmark comparison
make validate      # Quick validation
make clean         # Remove generated files
```

## Project Structure

```
nilm-vata/
├── src/
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── combinatorial_optimization.py  # CO algorithm
│   │   └── fhmm.py                        # FHMM algorithm
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── redd_loader.py                 # Dataset loader
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py               # Plotting functions
│       ├── metrics.py                     # Performance metrics
│       ├── model_persistence.py           # Save/load models
│       ├── energy_cost.py                 # Energy cost calculator
│       └── streaming.py                   # Real-time streaming
├── test_algorithms.py                     # Simple test script
├── demo_realtime.py                       # Real-time demo
├── benchmark.py                           # Algorithm benchmark tool
├── validate.py                            # Quick validation
├── pyproject.toml                         # Project configuration
├── requirements.txt                       # Python dependencies (legacy)
├── Makefile                               # Convenient commands
└── README.md                              # This file
```

## Usage Examples

### Basic Usage

```python
from src.datasets.redd_loader import REDDLoader
from src.algorithms.combinatorial_optimization import CombinatorialOptimization
from src.utils.metrics import calculate_metrics, print_metrics

# Load data
loader = REDDLoader()
train_data, test_data = loader.get_train_test_split(train_ratio=0.7)

# Train algorithm
co = CombinatorialOptimization(threshold=15.0)
co.train(train_data['mains'], train_data['appliances'])

# Disaggregate
predictions = co.disaggregate(test_data['mains'], num_samples=5000)

# Evaluate
metrics = calculate_metrics(test_data['appliances'], predictions)
print_metrics(metrics)
```

### Real-Time Processing

```python
from src.algorithms.fhmm import FHMM

# Train FHMM
fhmm = FHMM(n_states=2)
fhmm.train(train_data['mains'], train_data['appliances'])

# Process streaming data
for power_sample in streaming_data:
    chunk = np.array([power_sample]).reshape(-1, 1)
    predictions = fhmm.disaggregate_chunk(chunk)
    # predictions is a dict of {appliance: power_value}
```

### Visualization

```python
from src.utils.visualization import plot_disaggregation

# Plot disaggregation results
plot_disaggregation(
    mains=test_data['mains'],
    ground_truth=test_data['appliances'],
    predictions=predictions,
    start_idx=0,
    num_samples=1000,
    save_path='results.png'
)
```

## Advanced Features

### Model Persistence

Save and load trained models for reuse:

```python
from src.utils.model_persistence import ModelManager

# Initialize model manager
model_manager = ModelManager(models_dir='./models')

# Save a trained model
model_manager.save_model(
    model=co,
    model_name='my_co_model',
    algorithm_type='CO',
    metadata={'accuracy': 0.95, 'notes': 'Trained on REDD data'}
)

# List saved models
models = model_manager.list_models()
for model_info in models:
    print(f"Model: {model_info['name']}")
    print(f"Metadata: {model_info.get('metadata', {})}")

# Load a model
loaded_model = model_manager.load_model('./models/CO_my_co_model_20250118.pkl')
```

### Energy Cost Calculation

Calculate energy costs with time-of-use pricing:

```python
from src.utils.energy_cost import EnergyCostCalculator

# Initialize with pricing info
cost_calc = EnergyCostCalculator(
    rate_per_kwh=0.12,      # Standard rate
    peak_rate=0.18,          # Peak rate (optional)
    peak_hours=(17, 21)      # Peak period 5pm-9pm (optional)
)

# Calculate costs for appliances
costs = cost_calc.calculate_total_cost(
    appliances=test_data['appliances'],
    time_period='daily'  # or 'monthly', 'yearly'
)

# Generate detailed cost report with recommendations
report = cost_calc.generate_cost_report(
    appliances=test_data['appliances'],
    predictions=predictions  # Optional: include prediction accuracy
)
print(report)

# Visualize cost breakdown
cost_calc.plot_cost_breakdown(
    appliances=test_data['appliances'],
    save_path='cost_breakdown.png'
)
```

### Streaming Data Processing

Process data in real-time with async support:

```python
from src.utils.streaming import (
    DataStreamSimulator,
    StreamingDisaggregator,
    AsyncStreamProcessor
)

# Simulate a data stream
stream = DataStreamSimulator(
    data=test_data['mains'],
    sample_rate=1.0,  # 1 sample/second
    realtime=True
)

# Process with streaming disaggregator
disaggregator = StreamingDisaggregator(
    algorithm=co,
    window_size=100,
    update_interval=10
)

for chunk in stream.stream(chunk_size=1):
    predictions = disaggregator.process_sample(chunk[0])
    if predictions:
        print(f"Predictions: {predictions}")

# Or use async processor for non-blocking
def on_prediction(predictions):
    print(f"New predictions: {predictions}")

async_proc = AsyncStreamProcessor(
    algorithm=co,
    callback=on_prediction
)

async_proc.start()

# Add samples
for chunk in stream.stream(chunk_size=1):
    async_proc.add_sample(chunk[0])

# Get predictions
predictions = async_proc.get_predictions(block=True, timeout=1.0)

async_proc.stop()
```

## Algorithms

### Combinatorial Optimization (CO)

**How it works:**
- Learns ON/OFF power states for each appliance during training
- For each time step, finds the best combination of appliance states that matches the aggregate power
- Uses exhaustive search for small numbers of appliances, greedy approach for larger sets

**Pros:**
- Fast and deterministic
- Works well with distinct appliance power signatures
- No complex probability calculations

**Cons:**
- Assumes discrete ON/OFF states
- May struggle with continuously varying loads

### Factorial Hidden Markov Model (FHMM)

**How it works:**
- Models each appliance as a Hidden Markov Model
- Learns transition probabilities and emission distributions during training
- Uses Viterbi algorithm to infer most likely appliance states

**Pros:**
- Probabilistic approach handles uncertainty
- Can model multiple states per appliance
- Captures temporal dependencies

**Cons:**
- Slower than CO
- Requires more training data
- More parameters to tune

## Performance Metrics

The framework calculates several metrics:

- **MAE** (Mean Absolute Error): Average absolute difference in watts
- **RMSE** (Root Mean Squared Error): Square root of average squared error
- **NDE** (Normalized Disaggregation Error): Normalized energy estimation error
- **F1 Score**: Harmonic mean of precision and recall for ON/OFF detection
- **Energy Accuracy**: Percentage accuracy of total energy estimation

## Dataset

The current implementation uses **synthetic REDD-like data** that mimics the Reference Energy Disaggregation Dataset (REDD).

### Simulated Appliances

1. **Refrigerator**: Cyclic pattern (~10 min cycles, 200W when on)
2. **Microwave**: Sporadic high-power usage (1500W, 1-3 min)
3. **Light**: Time-based patterns (60W, morning and evening)

### Using Real REDD Data

To use the actual REDD dataset:

1. Download REDD from: http://redd.csail.mit.edu/
2. Extract to `./data/REDD/`
3. Modify `redd_loader.py` to read actual REDD files

## Extending the Framework

### Adding a New Algorithm

1. Create a new file in `src/algorithms/`
2. Implement the algorithm class with `train()` and `disaggregate()` methods
3. Add to `src/algorithms/__init__.py`
4. Update demo scripts to include the new algorithm

Example:

```python
class MyAlgorithm:
    def train(self, mains, appliances):
        # Training logic
        pass

    def disaggregate(self, mains, num_samples=None):
        # Disaggregation logic
        return predictions  # Dict of {appliance: DataFrame}
```

### Adding New Appliances

Modify the `download_sample_data()` method in `redd_loader.py`:

```python
# Add new appliance pattern
washing_machine = np.zeros(len(timestamps))
# ... define pattern ...
self.appliances['washing_machine'] = pd.DataFrame(
    {'power': washing_machine},
    index=timestamps
)
```

## Troubleshooting

### Import Errors

If you get import errors, ensure the `src` directory is in your Python path:

```python
import sys
sys.path.insert(0, 'src')
```

### Slow Training/Disaggregation

- Reduce the number of training/test samples
- Use CO instead of FHMM for faster processing
- For FHMM, reduce `n_iter` parameter in the HMM model

### Poor Performance

- Increase training data size
- Adjust threshold parameter for CO
- Use more states for FHMM (`n_states=3` or higher)
- Ensure appliances have distinct power signatures

## References

- **REDD**: Reference Energy Disaggregation Dataset
  - Kolter, J. Zico, and Matthew J. Johnson. "REDD: A public data set for energy disaggregation research." (2011)

- **NILMTK**: Non-Intrusive Load Monitoring Toolkit
  - Batra, Nipun, et al. "NILMTK: an open source toolkit for non-intrusive load monitoring." (2014)

- **FHMM**: Factorial Hidden Markov Models
  - Kolter, J. Zico, and Tommi Jaakkola. "Approximate inference in additive factorial HMMs with application to energy disaggregation." (2012)

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.
