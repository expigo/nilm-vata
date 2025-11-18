"""
REFIT Dataset Loader
Personalised Retrofit Decision Support Tools for the UK Housing Stock

Dataset Info:
- 20 houses in the UK
- 2-year monitoring period (2013-2015)
- 8 second sampling rate for aggregate
- Appliance-level data for many appliances
- Total size: ~500GB

Download: https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings


class REFITDataset:
    """
    REFIT dataset loader

    20 UK homes with detailed appliance-level measurements over 2 years.
    """

    # Common appliances across houses
    APPLIANCE_MAPPING = {
        'fridge': ['Fridge', 'Fridge Freezer'],
        'freezer': ['Freezer', 'Fridge Freezer'],
        'washer_dryer': ['Washing Machine', 'Washer Dryer', 'Tumble Dryer'],
        'dishwasher': ['Dishwasher'],
        'television': ['Television', 'TV'],
        'microwave': ['Microwave'],
        'kettle': ['Kettle'],
        'toaster': ['Toaster'],
        'computer': ['Computer', 'PC', 'Laptop'],
        'hi_fi': ['Hi-Fi', 'Audio System'],
        'electric_heater': ['Electric Heater', 'Electric heating'],
    }

    def __init__(self, dataset_path: str = './data/REFIT'):
        """
        Initialize REFIT dataset loader

        Args:
            dataset_path: Path to REFIT dataset directory
        """
        self.dataset_path = dataset_path

    def check_available(self) -> bool:
        """Check if dataset is available locally"""
        return os.path.exists(self.dataset_path) and \
               len([f for f in os.listdir(self.dataset_path) if f.startswith('CLEAN_House')]) > 0

    def download(self):
        """Provide download instructions"""
        print("="*70)
        print("REFIT Dataset Download Instructions")
        print("="*70)
        print("\nREFIT dataset must be downloaded manually:")
        print("\n1. Visit: https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements")
        print("2. Register and download the cleaned CSV files")
        print("3. Extract to:", self.dataset_path)
        print("\nExpected structure:")
        print("  ./data/REFIT/")
        print("    ├── CLEAN_House1.csv")
        print("    ├── CLEAN_House2.csv")
        print("    └── ...")
        print("\nDataset size: ~500GB")
        print("Alternative: Download individual houses to save space")
        print("="*70)

    def load_building(
        self,
        building_id: int = 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        appliances: Optional[List[str]] = None
    ) -> Dict:
        """
        Load data from a specific house

        Args:
            building_id: House number (1-20)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            appliances: List of appliances to load

        Returns:
            Dictionary with 'mains' and 'appliances' data
        """
        if not self.check_available():
            print("Dataset not found. Please download REFIT first.")
            self.download()
            raise FileNotFoundError(f"REFIT dataset not found at {self.dataset_path}")

        house_file = os.path.join(self.dataset_path, f'CLEAN_House{building_id}.csv')

        if not os.path.exists(house_file):
            raise FileNotFoundError(f"House {building_id} not found at {house_file}")

        print(f"Loading REFIT House {building_id}...")

        # Read CSV
        df = pd.read_csv(house_file, parse_dates=['Time'], index_col='Time')

        # Filter by date if specified
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        # Extract mains (aggregate)
        mains = df['Aggregate'].copy()
        mains_df = pd.DataFrame({'power': mains})

        # Extract appliances
        appliances_data = {}
        available_appliances = [col for col in df.columns if col != 'Aggregate' and col != 'Issues']

        for col in available_appliances:
            # Map to standardized names
            standardized_name = self._standardize_appliance_name(col)

            if appliances is None or standardized_name in appliances:
                appliances_data[standardized_name] = pd.DataFrame({
                    'power': df[col]
                })
                print(f"  Loaded {standardized_name} ({col}): {len(df[col])} samples")

        return {
            'mains': mains_df,
            'appliances': appliances_data
        }

    def _standardize_appliance_name(self, name: str) -> str:
        """Convert REFIT appliance name to standardized format"""
        name_lower = name.lower()

        for std_name, variants in self.APPLIANCE_MAPPING.items():
            for variant in variants:
                if variant.lower() in name_lower:
                    return std_name

        # Return cleaned version of original name
        return name.lower().replace(' ', '_').replace('-', '_')

    def get_train_test_split(
        self,
        building_id: int = 1,
        train_ratio: float = 0.7,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Load building and split into train/test

        Args:
            building_id: House number
            train_ratio: Ratio for training data
            **kwargs: Additional arguments for load_building

        Returns:
            Tuple of (train_data, test_data)
        """
        data = self.load_building(building_id, **kwargs)

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

    def list_available_houses(self) -> List[int]:
        """List available house IDs"""
        if not self.check_available():
            return []

        files = os.listdir(self.dataset_path)
        houses = []

        for f in files:
            if f.startswith('CLEAN_House') and f.endswith('.csv'):
                try:
                    house_num = int(f.replace('CLEAN_House', '').replace('.csv', ''))
                    houses.append(house_num)
                except:
                    pass

        return sorted(houses)
