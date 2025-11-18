"""
ECO Dataset Loader
Electricity Consumption & Occupancy

Dataset Info:
- 6 households in Switzerland
- 8 months of data
- 1 Hz sampling rate for aggregate
- Occupancy data from sensors
- Smart plugs for appliance-level data

Download: https://www.vs.inf.ethz.ch/res/show.html?what=eco-data
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional


class ECODataset:
    """
    ECO (Electricity Consumption & Occupancy) dataset loader

    6 Swiss homes with high-frequency monitoring and occupancy data.
    """

    def __init__(self, dataset_path: str = './data/ECO'):
        """
        Initialize ECO dataset loader

        Args:
            dataset_path: Path to ECO dataset directory
        """
        self.dataset_path = dataset_path

    def check_available(self) -> bool:
        """Check if dataset is available locally"""
        # Check for at least one house directory
        for i in range(1, 7):
            house_path = os.path.join(self.dataset_path, f'{i:02d}')
            if os.path.exists(house_path):
                return True
        return False

    def download(self):
        """Provide download instructions"""
        print("="*70)
        print("ECO Dataset Download Instructions")
        print("="*70)
        print("\nECO dataset must be downloaded manually:")
        print("\n1. Visit: https://www.vs.inf.ethz.ch/res/show.html?what=eco-data")
        print("2. Download the dataset (requires email registration)")
        print("3. Extract to:", self.dataset_path)
        print("\nExpected structure:")
        print("  ./data/ECO/")
        print("    ├── 01/  (House 1)")
        print("    │   ├── 01_sm.csv  (Smart meter - aggregate)")
        print("    │   ├── 01_plugs.csv  (Smart plugs - appliances)")
        print("    │   └── ...")
        print("    ├── 02/  (House 2)")
        print("    └── ...")
        print("\nDataset size: ~3GB")
        print("Features: 1Hz sampling, occupancy data")
        print("="*70)

    def load_building(
        self,
        building_id: int = 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Load data from a specific house

        Args:
            building_id: House number (1-6)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary with 'mains' and 'appliances' data
        """
        if not self.check_available():
            print("Dataset not found. Please download ECO first.")
            self.download()
            raise FileNotFoundError(f"ECO dataset not found at {self.dataset_path}")

        house_path = os.path.join(self.dataset_path, f'{building_id:02d}')

        if not os.path.exists(house_path):
            raise FileNotFoundError(f"House {building_id} not found at {house_path}")

        print(f"Loading ECO House {building_id:02d}...")

        # Load smart meter (aggregate)
        sm_file = os.path.join(house_path, f'{building_id:02d}_sm.csv')
        if not os.path.exists(sm_file):
            raise FileNotFoundError(f"Smart meter file not found: {sm_file}")

        sm_df = pd.read_csv(sm_file, parse_dates=[0], index_col=0)

        # Filter by date
        if start_date:
            sm_df = sm_df[sm_df.index >= start_date]
        if end_date:
            sm_df = sm_df[sm_df.index <= end_date]

        # Aggregate is sum of all phases
        mains_power = sm_df.sum(axis=1)
        mains_df = pd.DataFrame({'power': mains_power})

        # Load smart plugs (appliances)
        plugs_file = os.path.join(house_path, f'{building_id:02d}_plugs.csv')
        appliances_data = {}

        if os.path.exists(plugs_file):
            plugs_df = pd.read_csv(plugs_file, parse_dates=[0], index_col=0)

            # Filter by date
            if start_date:
                plugs_df = plugs_df[plugs_df.index >= start_date]
            if end_date:
                plugs_df = plugs_df[plugs_df.index <= end_date]

            # Each column is an appliance/plug
            for col in plugs_df.columns:
                # Clean column name
                app_name = col.lower().replace(' ', '_').replace('-', '_')
                appliances_data[app_name] = pd.DataFrame({'power': plugs_df[col]})
                print(f"  Loaded {app_name}: {len(plugs_df[col])} samples")

        return {
            'mains': mains_df,
            'appliances': appliances_data
        }

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
