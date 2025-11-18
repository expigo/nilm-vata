"""
Real NILM dataset loaders (REDD, UK-DALE, etc.)

Provides utilities to download and load popular NILM datasets.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings

try:
    import requests
    from tqdm import tqdm
    DOWNLOAD_AVAILABLE = True
except ImportError:
    DOWNLOAD_AVAILABLE = False
    warnings.warn("Install requests and tqdm for dataset download: pip install requests tqdm")


class REDDDataset:
    """
    REDD (Reference Energy Disaggregation Dataset) loader

    REDD is a public dataset for energy disaggregation research,
    containing data from 6 homes with labeled appliances.

    Dataset info: http://redd.csail.mit.edu/
    """

    APPLIANCE_CHANNELS = {
        'fridge': [5],
        'microwave': [11],
        'washer_dryer': [10],
        'dishwasher': [9],
        'lighting': [3, 4],
        'electronics': [6, 7],
        'kitchen_outlets': [8],
    }

    def __init__(self, dataset_path: str = './data/REDD'):
        """
        Initialize REDD dataset loader

        Args:
            dataset_path: Path to REDD dataset directory
        """
        self.dataset_path = dataset_path
        self.data = {}

    def check_available(self) -> bool:
        """Check if dataset is available locally"""
        return os.path.exists(self.dataset_path) and \
               os.path.exists(os.path.join(self.dataset_path, 'low_freq'))

    def download(self):
        """
        Inform user how to download REDD

        Note: REDD requires manual download due to licensing
        """
        print("="*70)
        print("REDD Dataset Download Instructions")
        print("="*70)
        print("\nREDD dataset must be downloaded manually:")
        print("\n1. Visit: http://redd.csail.mit.edu/")
        print("2. Download the dataset (low frequency recommended)")
        print("3. Extract to:", self.dataset_path)
        print("\nExpected structure:")
        print("  ./data/REDD/")
        print("    └── low_freq/")
        print("        ├── house_1/")
        print("        ├── house_2/")
        print("        └── ...")
        print("\nAlternatively, you can use the synthetic REDD-like data")
        print("by using the REDDLoader class instead.")
        print("="*70)

    def load_building(
        self,
        building_id: int = 1,
        appliances: Optional[List[str]] = None
    ) -> Dict:
        """
        Load data from a specific building

        Args:
            building_id: Building number (1-6 for REDD)
            appliances: List of appliances to load (None = all)

        Returns:
            Dictionary with 'mains' and 'appliances' data
        """
        if not self.check_available():
            print("Dataset not found. Please download REDD first.")
            self.download()
            raise FileNotFoundError(f"REDD dataset not found at {self.dataset_path}")

        house_path = os.path.join(self.dataset_path, 'low_freq', f'house_{building_id}')

        if not os.path.exists(house_path):
            raise FileNotFoundError(f"House {building_id} not found at {house_path}")

        print(f"Loading REDD House {building_id}...")

        # Load mains (channel 1 and 2)
        mains_1 = self._load_channel(house_path, 1)
        mains_2 = self._load_channel(house_path, 2)

        if mains_1 is not None and mains_2 is not None:
            # Combine both mains
            mains = mains_1.add(mains_2, fill_value=0)
        elif mains_1 is not None:
            mains = mains_1
        elif mains_2 is not None:
            mains = mains_2
        else:
            raise ValueError("No mains data found")

        mains_df = pd.DataFrame({'power': mains})

        # Load appliances
        if appliances is None:
            appliances = list(self.APPLIANCE_CHANNELS.keys())

        appliances_data = {}

        for app_name in appliances:
            if app_name not in self.APPLIANCE_CHANNELS:
                print(f"  Warning: Unknown appliance '{app_name}', skipping")
                continue

            channels = self.APPLIANCE_CHANNELS[app_name]
            app_data = None

            for channel in channels:
                channel_data = self._load_channel(house_path, channel)

                if channel_data is not None:
                    if app_data is None:
                        app_data = channel_data
                    else:
                        app_data = app_data.add(channel_data, fill_value=0)

            if app_data is not None:
                appliances_data[app_name] = pd.DataFrame({'power': app_data})
                print(f"  Loaded {app_name}: {len(app_data)} samples")

        return {
            'mains': mains_df,
            'appliances': appliances_data
        }

    def _load_channel(self, house_path: str, channel: int) -> Optional[pd.Series]:
        """Load data from a specific channel"""
        channel_file = os.path.join(house_path, f'channel_{channel}.dat')

        if not os.path.exists(channel_file):
            return None

        try:
            # REDD format: timestamp power
            data = pd.read_csv(
                channel_file,
                sep=' ',
                names=['timestamp', 'power'],
                dtype={'timestamp': int, 'power': float}
            )

            # Convert timestamp to datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
            data = data.set_index('timestamp')

            return data['power']

        except Exception as e:
            print(f"  Warning: Failed to load channel {channel}: {e}")
            return None

    def get_train_test_split(
        self,
        building_id: int = 1,
        train_ratio: float = 0.7,
        appliances: Optional[List[str]] = None
    ) -> Tuple[Dict, Dict]:
        """
        Load building and split into train/test

        Args:
            building_id: Building number
            train_ratio: Ratio for training data
            appliances: List of appliances to include

        Returns:
            Tuple of (train_data, test_data)
        """
        data = self.load_building(building_id, appliances)

        split_idx = int(len(data['mains']) * train_ratio)

        train_data = {
            'mains': data['mains'].iloc[:split_idx],
            'appliances': {
                name: df.iloc[:split_idx]
                for name, df in data['appliances'].items()
            }
        }

        test_data = {
            'mains': data['mains'].iloc[split_idx:],
            'appliances': {
                name: df.iloc[split_idx:]
                for name, df in data['appliances'].items()
            }
        }

        print(f"Train samples: {len(train_data['mains']):,}")
        print(f"Test samples: {len(test_data['mains']):,}")

        return train_data, test_data


class UKDALEDataset:
    """
    UK-DALE (UK Domestic Appliance-Level Electricity) dataset loader

    UK-DALE contains data from 5 UK homes with detailed appliance labeling.

    Dataset info: https://jack-kelly.com/data/
    """

    def __init__(self, dataset_path: str = './data/UKDALE'):
        """
        Initialize UK-DALE dataset loader

        Args:
            dataset_path: Path to UK-DALE dataset directory
        """
        self.dataset_path = dataset_path

    def check_available(self) -> bool:
        """Check if dataset is available locally"""
        return os.path.exists(self.dataset_path)

    def download(self):
        """
        Inform user how to download UK-DALE

        Note: UK-DALE requires manual download
        """
        print("="*70)
        print("UK-DALE Dataset Download Instructions")
        print("="*70)
        print("\nUK-DALE dataset must be downloaded manually:")
        print("\n1. Visit: https://jack-kelly.com/data/")
        print("2. Download the dataset (HDF5 format recommended)")
        print("3. Extract to:", self.dataset_path)
        print("\nExpected structure:")
        print("  ./data/UKDALE/")
        print("    ├── ukdale.h5  (or)")
        print("    ├── house_1.h5")
        print("    ├── house_2.h5")
        print("    └── ...")
        print("\nNote: UK-DALE requires the 'tables' package:")
        print("  pip install tables")
        print("="*70)

    def load_building(self, building_id: int = 1) -> Dict:
        """
        Load data from a specific building

        Args:
            building_id: Building number (1-5 for UK-DALE)

        Returns:
            Dictionary with 'mains' and 'appliances' data
        """
        print("UK-DALE loader not fully implemented yet.")
        print("Using synthetic data instead.")
        self.download()

        raise NotImplementedError(
            "UK-DALE loader requires HDF5 support. "
            "Please use REDDDataset or REDDLoader (synthetic) instead."
        )


class DatasetRegistry:
    """Registry of available NILM datasets"""

    DATASETS = {
        'redd': REDDDataset,
        'ukdale': UKDALEDataset,
    }

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all available dataset loaders"""
        return list(cls.DATASETS.keys())

    @classmethod
    def get_dataset(cls, name: str, **kwargs):
        """
        Get a dataset loader by name

        Args:
            name: Dataset name ('redd', 'ukdale', etc.)
            **kwargs: Arguments for dataset constructor

        Returns:
            Dataset loader instance
        """
        if name.lower() not in cls.DATASETS:
            raise ValueError(
                f"Unknown dataset: {name}. "
                f"Available: {', '.join(cls.list_datasets())}"
            )

        return cls.DATASETS[name.lower()](**kwargs)

    @classmethod
    def check_availability(cls) -> Dict[str, bool]:
        """Check which datasets are available locally"""
        availability = {}

        for name, dataset_class in cls.DATASETS.items():
            try:
                dataset = dataset_class()
                availability[name] = dataset.check_available()
            except:
                availability[name] = False

        return availability
