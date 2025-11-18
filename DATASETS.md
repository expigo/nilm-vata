# NILM Datasets Guide

Comprehensive guide to all NILM (Non-Intrusive Load Monitoring) datasets supported by NILM VATA.

## 📊 Dataset Overview

NILM VATA supports **11 real-world datasets** + synthetic data, covering different countries, sampling rates, and use cases.

### Quick Comparison

| Dataset | Homes | Country | Duration | Sampling | Size | Status |
|---------|-------|---------|----------|----------|------|--------|
| **REDD** | 6 | USA | 3-19 days | 15 kHz / 1 Hz | ~10GB | ✅ Loader Ready |
| **UK-DALE** | 5 | UK | 4+ years | 16 kHz / 6s | ~800GB | ✅ Loader Ready |
| **REFIT** | 20 | UK | 2 years | 8s | ~500GB | ✅ Loader Ready |
| **AMPds** | 1 | Canada | 2 years | 1 min | ~12GB | ✅ Loader Ready |
| **ECO** | 6 | Switzerland | 8 months | 1 Hz | ~3GB | ✅ Loader Ready |
| **GREEND** | 9 | Italy/Austria | 1 year | 1 Hz | ~2GB | 🔧 In Progress |
| **iAWE** | 1 | India | 73 days | 1 Hz | ~1GB | 🔧 In Progress |
| **DRED** | 1 | Netherlands | 6 months | 1 Hz | ~300MB | 🔧 In Progress |
| **PLAID** | - | USA | - | 30 kHz | ~400MB | ⚡ High-freq only |
| **BLUED** | 1 | USA | 8 days | 12 kHz | ~50GB | ⚡ High-freq only |
| **Synthetic** | ∞ | - | Custom | 1s | - | ✅ Built-in |

**Legend:**
- ✅ = Fully implemented and tested
- 🔧 = Loader structure ready, implementation in progress
- ⚡ = High-frequency only (for event detection, not typical disaggregation)

---

## 📚 Detailed Dataset Information

### 1. REDD (Reference Energy Disaggregation Dataset) 🇺🇸

**Best for:** Benchmarking, algorithm comparison

```python
from src.datasets import REDDDataset

dataset = REDDDataset(dataset_path='./data/REDD')
train_data, test_data = dataset.get_train_test_split(building_id=1)
```

**Details:**
- **Homes:** 6 houses in Massachusetts, USA
- **Duration:** 3-19 days per house
- **Sampling:** 15 kHz (high-freq) + 1 Hz (low-freq)
- **Appliances:** 10-20 per house (labeled)
- **Size:** ~10GB
- **Download:** http://redd.csail.mit.edu/

**Pros:**
- Most commonly used benchmark
- Well-documented and clean
- Multiple sampling rates

**Cons:**
- Short duration (days not months)
- Only 6 homes
- USA-specific appliances

---

### 2. UK-DALE (UK Domestic Appliance-Level Electricity) 🇬🇧

**Best for:** Long-term patterns, seasonal analysis

```python
from src.datasets import UKDALEDataset

dataset = UKDALEDataset(dataset_path='./data/UKDALE')
# Implementation uses HDF5 format
```

**Details:**
- **Homes:** 5 houses in UK
- **Duration:** Up to 4+ years
- **Sampling:** 16 kHz (high-freq) + 6 seconds (low-freq)
- **Appliances:** Multiple per house with detailed labeling
- **Size:** ~800GB (full dataset)
- **Download:** https://jack-kelly.com/data/

**Pros:**
- Longest duration dataset (years!)
- UK electricity patterns
- Very detailed labeling

**Cons:**
- HUGE dataset size
- Requires HDF5 (pytables)
- Long download time

**Recommendation:** Download only specific houses or time periods to save space.

---

### 3. REFIT 🇬🇧

**Best for:** Large-scale studies, diverse households

```python
from src.datasets import REFITDataset

dataset = REFITDataset(dataset_path='./data/REFIT')
train_data, test_data = dataset.get_train_test_split(building_id=1)
```

**Details:**
- **Homes:** 20 houses in UK
- **Duration:** 2 years (2013-2015)
- **Sampling:** 8 seconds
- **Appliances:** Many per house (fridge, washing machine, kettle, etc.)
- **Size:** ~500GB
- **Download:** https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements

**Pros:**
- Most homes (20!)
- Long duration (2 years)
- Cleaned CSV format

**Cons:**
- Very large dataset
- UK-specific appliances
- Manual download required

**Tip:** Download individual houses to save space.

---

### 4. AMPds (Almanac of Minutely Power dataset) 🇨🇦

**Best for:** Single-home detailed study, weather correlation

```python
from src.datasets import AMPdsDataset

dataset = AMPdsDataset(dataset_path='./data/AMPds')
data = dataset.load_building(building_id=1, include_weather=True)
```

**Details:**
- **Homes:** 1 house in British Columbia, Canada
- **Duration:** 2 years (2012-2014)
- **Sampling:** 1 minute
- **Appliances:** 21 circuits/appliances
- **Extra:** Water consumption + weather data
- **Size:** ~12GB
- **Download:** http://ampds.org/

**Pros:**
- Includes weather data!
- Also has water monitoring
- Very detailed for single home
- Organized by circuit

**Cons:**
- Only 1 home
- Requires registration
- Canada-specific patterns

**Unique feature:** Weather correlation analysis possible!

---

### 5. ECO (Electricity Consumption & Occupancy) 🇨🇭

**Best for:** Occupancy-aware disaggregation

```python
from src.datasets import ECODataset

dataset = ECODataset(dataset_path='./data/ECO')
train_data, test_data = dataset.get_train_test_split(building_id=1)
```

**Details:**
- **Homes:** 6 households in Switzerland
- **Duration:** 8 months
- **Sampling:** 1 Hz (aggregate) + smart plugs
- **Extra:** Occupancy sensors!
- **Size:** ~3GB
- **Download:** https://www.vs.inf.ethz.ch/res/show.html?what=eco-data

**Pros:**
- High sampling rate (1 Hz)
- Occupancy data included
- Manageable size
- Smart plug ground truth

**Cons:**
- Only 8 months
- Swiss electricity patterns
- Requires email registration

**Unique feature:** Occupancy sensors for context-aware NILM!

---

### 6. GREEND 🇮🇹 🇦🇹

**Best for:** European patterns, cross-country comparison

```python
from src.datasets import GREENDDataset

dataset = GREENDDataset(dataset_path='./data/GREEND')
# Implementation in progress
```

**Details:**
- **Homes:** 9 households (8 Austria, 1 Italy)
- **Duration:** 1 year
- **Sampling:** 1 Hz
- **Size:** ~2GB
- **Download:** https://sourceforge.net/projects/greend/

**Pros:**
- Multiple countries
- High sampling rate
- Reasonable size

**Cons:**
- Limited documentation
- Mixed data quality

---

### 7. iAWE 🇮🇳

**Best for:** Indian electricity patterns, water correlation

```python
from src.datasets import iAWEDataset

dataset = iAWEDataset(dataset_path='./data/iAWE')
# Implementation in progress
```

**Details:**
- **Homes:** 1 household in India
- **Duration:** 73 days
- **Sampling:** 1 Hz
- **Extra:** Water consumption
- **Size:** ~1GB
- **Download:** http://iawe.github.io/

**Pros:**
- Indian appliance patterns
- Water data included
- High sampling rate
- Small size

**Cons:**
- Only 1 home
- Short duration

**Unique feature:** First major Indian NILM dataset!

---

### 8. DRED (Dutch Residential Energy Dataset) 🇳🇱

**Best for:** Dutch/European patterns

```python
from src.datasets import DREDDataset

dataset = DREDDataset(dataset_path='./data/DRED')
# Implementation in progress
```

**Details:**
- **Homes:** 1 household in Netherlands
- **Duration:** 6 months
- **Sampling:** 1 Hz
- **Size:** ~300MB
- **Download:** http://www.st.ewi.tudelft.nl/~akshay/dred/

**Pros:**
- Small size (easy download)
- High sampling rate
- Well-documented

**Cons:**
- Only 1 home
- Moderate duration

---

### 9. PLAID ⚡

**Type:** HIGH-FREQUENCY (Event Detection)

```python
from src.datasets import PLAIDDataset

dataset = PLAIDDataset(dataset_path='./data/PLAID')
# For appliance identification, not disaggregation
```

**Details:**
- **Type:** Appliance signature database
- **Instances:** 1000+ appliance instances
- **Types:** 11 appliance types
- **Sampling:** 30 kHz (!!)
- **Size:** ~400MB
- **Download:** https://energy.duke.edu/content/plaid-plug-load-appliance-identification-dataset

**Use case:** Appliance IDENTIFICATION (not disaggregation)

**Good for:**
- Training appliance classifiers
- Transient event detection
- Appliance signature analysis

**Not for:** Continuous power disaggregation

---

### 10. BLUED ⚡

**Type:** HIGH-FREQUENCY (Event Detection)

```python
from src.datasets import BLUEDDataset

dataset = BLUEDDataset(dataset_path='./data/BLUED')
# For event detection
```

**Details:**
- **Homes:** 1 residence in USA
- **Duration:** 8 days
- **Sampling:** 12 kHz (!!)
- **Labels:** Fully labeled events
- **Size:** ~50GB
- **Download:** http://portoalegre.andrew.cmu.edu:88/BLUED/

**Use case:** Event DETECTION (not disaggregation)

**Good for:**
- Event detection algorithms
- Transient analysis
- High-frequency patterns

**Not for:** Typical NILM disaggregation (too high frequency)

---

### 11. Synthetic REDD-like Data (Built-in) ✨

**Best for:** Quick testing, development, demos

```python
from src.datasets import REDDLoader

loader = REDDLoader()  # No download needed!
train_data, test_data = loader.get_train_test_split()
```

**Details:**
- **Homes:** Unlimited (generated on demand)
- **Duration:** Customizable
- **Sampling:** 1 second
- **Appliances:** Fridge, microwave, light (more can be added)
- **Size:** Generated in memory
- **Download:** NONE - built-in!

**Pros:**
- ✅ No download required
- ✅ Instant availability
- ✅ Perfect for testing
- ✅ Customizable

**Cons:**
- Simplified appliance models
- Not real-world patterns
- Limited appliance types

**Perfect for:** Algorithm development, quick demos, tutorials

---

## 🎯 Which Dataset Should I Use?

### For Algorithm Development & Testing
→ **Synthetic REDD** (instant, no download)

### For Benchmarking
→ **REDD** (most common benchmark)

### For Long-term Patterns
→ **UK-DALE** or **REFIT** (years of data)

### For Weather Correlation
→ **AMPds** (includes weather)

### For Occupancy-aware NILM
→ **ECO** (includes occupancy)

### For Large-scale Studies
→ **REFIT** (20 homes)

### For Cross-country Comparison
→ **GREEND** (Italy + Austria)

### For Appliance Identification
→ **PLAID** (high-freq signatures)

### For Event Detection
→ **BLUED** (high-freq with labels)

---

## 📥 Installation Guide

### Quick Start (No Real Data)
```bash
# Use synthetic data - works out of the box!
python validate.py
```

### Download Real Datasets

**Option 1: Manual Download**
1. Visit dataset website
2. Download and extract
3. Place in `./data/<DATASET_NAME>/`

**Option 2: Automated (Coming Soon)**
```bash
# Future feature
make download-redd
make download-ukdale
```

### Install Dataset Dependencies
```bash
# For HDF5 support (UK-DALE, etc.)
uv pip install -e ".[datasets]"
```

---

## 🔍 Dataset Registry

Check which datasets are available:

```python
from src.datasets import DatasetRegistry

# List all supported datasets
datasets = DatasetRegistry.list_datasets()
print(datasets)

# Check what's available locally
availability = DatasetRegistry.check_availability()
for name, available in availability.items():
    status = "✓ Ready" if available else "✗ Not found"
    print(f"{name}: {status}")
```

---

## 📊 Recommended Dataset Combinations

### Starter Pack (< 20GB)
- REDD (10GB) - Standard benchmark
- AMPds (12GB) - Detailed single-home

### Research Pack (< 100GB)
- REDD (10GB)
- REFIT - 5 houses (125GB total)
- ECO (3GB)

### Complete Pack (< 1TB)
- All low-frequency datasets
- Synthetic for quick testing

---

## 🌍 Geographic Coverage

- 🇺🇸 **USA:** REDD, PLAID, BLUED
- 🇬🇧 **UK:** UK-DALE, REFIT
- 🇨🇦 **Canada:** AMPds
- 🇨🇭 **Switzerland:** ECO
- 🇮🇹 **Italy:** GREEND (1 home)
- 🇦🇹 **Austria:** GREEND (8 homes)
- 🇮🇳 **India:** iAWE
- 🇳🇱 **Netherlands:** DRED

---

## 📝 Dataset Citations

**REDD:**
```
Kolter, J. Zico, and Matthew J. Johnson. "REDD: A public data set for energy disaggregation research."
Workshop on Data Mining Applications in Sustainability (SIGKDD), 2011.
```

**UK-DALE:**
```
Kelly, Jack, and William Knottenbelt. "The UK-DALE dataset, domestic appliance-level electricity demand and whole-house demand from five UK homes."
Scientific Data 2.1 (2015): 1-14.
```

**REFIT:**
```
Murray, David, et al. "A data management platform for personalised real-time energy feedback."
Proceedings of the 8th International Conference on Energy Efficiency in Domestic Appliances and Lighting. 2015.
```

**AMPds:**
```
Makonin, Stephen, et al. "Exploiting HMM sparsity to perform online real-time nonintrusive load monitoring."
IEEE Transactions on Smart Grid 7.6 (2015): 2575-2585.
```

**ECO:**
```
Beckel, Christian, et al. "The ECO data set and the performance of non-intrusive load monitoring algorithms."
Proceedings of the 1st ACM Conference on Embedded Systems for Energy-Efficient Buildings. 2014.
```

---

## 🚀 Next Steps

1. **Start with synthetic data** - No download needed
2. **Download REDD** - Standard benchmark (10GB)
3. **Try other datasets** - Based on your research needs
4. **Benchmark your algorithms** - Compare across datasets

```bash
# Test with synthetic data (instant)
python validate.py

# Run showcase with any dataset
python showcase_algorithms.py

# Benchmark across datasets
python benchmark.py
```

---

## 📞 Support

For dataset-specific issues:
- Check dataset website for updates
- Verify download integrity
- Ensure correct file structure
- Open issue on GitHub

---

**Total Datasets Supported:** 11 real + 1 synthetic = **12 datasets**

**Geographic Coverage:** 8 countries across 4 continents

**Total Available Data:** 1+ TB (if you download everything!)
