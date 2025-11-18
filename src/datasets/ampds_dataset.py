"""
AMPds Dataset Loader
Almanac of Minutely Power dataset

Dataset Info:
- 1 house in Canada
- 2 years of data (2012-2014)
- 1-minute sampling rate
- Electricity and water monitoring
- 21 individual circuits/appliances
- Weather data included

Download: http://ampds.org/
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional


class AMPdsDataset:
    """
    AMPds (Almanac of Minutely Power dataset) loader

    Single home in Canada with detailed monitoring at 1-minute resolution.
    """

    # Circuit/appliance mapping
    APPLIANCES = {
        'fridge': 'FRE',
        'dishwasher': 'DWE',
        'furnace': 'FGE',
        'garage_door': 'GRE',
        'microwave': 'MWE',
        'washer': 'CWE',
        'dryer': 'DYE',
        'electric_heat': 'EHE',
        'kitchen_outlets': 'B1E',
        'living_room_outlets': 'B2E',
        'bedroom_outlets': 'BME',
        'electronics': 'EBE',
        'outdoor_outlets': 'OFE',
        'lighting': 'OUE',
        'utility_room': 'UTE',
    }

    def __init__(self, dataset_path: str = './data/AMPds'):
        """
        Initialize AMPds dataset loader

        Args:
            dataset_path: Path to AMPds dataset directory
        """
        self.dataset_path = dataset_path

    def check_available(self) -> bool:
        """Check if dataset is available locally"""
        electricity_file = os.path.join(self.dataset_path, 'electricity', 'WHE.csv')
        return os.path.exists(electricity_file)

    def download(self):
        """Provide download instructions"""
        print("="*70)
        print("AMPds Dataset Download Instructions")
        print("="*70)
        print("\nAMPds dataset must be downloaded manually:")
        print("\n1. Visit: http://ampds.org/")
        print("2. Register for academic use")
        print("3. Download the dataset")
        print("4. Extract to:", self.dataset_path)
        print("\nExpected structure:")
        print("  ./data/AMPds/")
        print("    ├── electricity/")
        print("    │   ├── WHE.csv (Whole House Electricity)")
        print("    │   ├── FRE.csv (Fridge)")
        print("    │   ├── DWE.csv (Dishwasher)")
        print("    │   └── ...")
        print("    ├── water/")
        print("    └── weather/")
        print("\nDataset size: ~12GB")
        print("="*70)

    def load_building(
        self,
        building_id: int = 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        appliances: Optional[List[str]] = None,
        include_weather: bool = False
    ) -> Dict:
        """
        Load AMPds data (single home)

        Args:
            building_id: Always 1 (single house dataset)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            appliances: List of appliances to load
            include_weather: Include weather data

        Returns:
            Dictionary with 'mains' and 'appliances' data
        """
        if not self.check_available():
            print("Dataset not found. Please download AMPds first.")
            self.download()
            raise FileNotFoundError(f"AMPds dataset not found at {self.dataset_path}")

        print(f"Loading AMPds...")

        electricity_path = os.path.join(self.dataset_path, 'electricity')

        # Load whole house electricity (mains)
        whe_file = os.path.join(electricity_path, 'WHE.csv')
        mains_df = pd.read_csv(whe_file, parse_dates=['TS'], index_col='TS')

        # Filter by date if specified
        if start_date:
            mains_df = mains_df[mains_df.index >= start_date]
        if end_date:
            mains_df = mains_df[mains_df.index <= end_date]

        mains_data = pd.DataFrame({'power': mains_df.iloc[:, 0]})  # First column is active power

        # Load appliances
        appliances_data = {}

        if appliances is None:
            appliances = list(self.APPLIANCES.keys())

        for app_name in appliances:
            if app_name not in self.APPLIANCES:
                print(f"  Warning: Unknown appliance '{app_name}', skipping")
                continue

            code = self.APPLIANCES[app_name]
            app_file = os.path.join(electricity_path, f'{code}.csv')

            if not os.path.exists(app_file):
                print(f"  Warning: File not found for {app_name} ({code}), skipping")
                continue

            app_df = pd.read_csv(app_file, parse_dates=['TS'], index_col='TS')

            # Filter by date
            if start_date:
                app_df = app_df[app_df.index >= start_date]
            if end_date:
                app_df = app_df[app_df.index <= end_date]

            appliances_data[app_name] = pd.DataFrame({'power': app_df.iloc[:, 0]})
            print(f"  Loaded {app_name}: {len(app_df)} samples")

        result = {
            'mains': mains_data,
            'appliances': appliances_data
        }

        # Optionally load weather
        if include_weather:
            weather_file = os.path.join(self.dataset_path, 'weather', 'Weather.csv')
            if os.path.exists(weather_file):
                weather_df = pd.read_csv(weather_file, parse_dates=['TS'], index_col='TS')
                if start_date:
                    weather_df = weather_df[weather_df.index >= start_date]
                if end_date:
                    weather_df = weather_df[weather_df.index <= end_date]
                result['weather'] = weather_df
                print(f"  Loaded weather data: {len(weather_df)} samples")

        return result

    def get_train_test_split(
        self,
        building_id: int = 1,
        train_ratio: float = 0.7,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Load and split into train/test

        Args:
            building_id: Always 1
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
