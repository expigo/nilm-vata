"""
REDD Dataset Loader
Loads and prepares REDD dataset for NILM algorithms
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class REDDLoader:
    """Load and prepare REDD dataset for NILM algorithms"""

    def __init__(self, dataset_path: str = './data/REDD'):
        """
        Initialize REDD dataset loader

        Args:
            dataset_path: Path to REDD dataset directory
        """
        self.dataset_path = dataset_path
        self.buildings = {}
        self.mains = None
        self.appliances = {}

    def download_sample_data(self):
        """
        Create sample REDD-like data for testing
        This generates synthetic data mimicking REDD structure
        """
        print("Generating sample REDD-like data for testing...")
        os.makedirs(self.dataset_path, exist_ok=True)

        # Generate synthetic aggregate and appliance data
        # Simulate 24 hours of data at 1-second sampling
        timestamps = pd.date_range('2025-01-01', periods=86400, freq='1s')

        # Create realistic appliance patterns
        # Refrigerator: cyclic pattern (40W on average, cycles every ~10 mins)
        fridge = np.zeros(len(timestamps))
        for i in range(0, len(timestamps), 600):  # 10-minute cycles
            duration = np.random.randint(180, 300)  # 3-5 minute on-time
            end_idx = min(i + duration, len(timestamps))
            fridge[i:end_idx] = np.random.normal(200, 10, end_idx - i)

        # Microwave: sporadic high power usage
        microwave = np.zeros(len(timestamps))
        usage_times = np.random.choice(len(timestamps), 5, replace=False)
        for start_idx in usage_times:
            duration = np.random.randint(60, 180)  # 1-3 minutes
            end_idx = min(start_idx + duration, len(timestamps))
            microwave[start_idx:end_idx] = np.random.normal(1500, 50, end_idx - start_idx)

        # Light: multiple on/off periods during the day
        light = np.zeros(len(timestamps))
        # Morning (6-8 AM)
        morning_start = 6 * 3600
        morning_end = 8 * 3600
        light[morning_start:morning_end] = np.random.normal(60, 5, morning_end - morning_start)
        # Evening (6-11 PM)
        evening_start = 18 * 3600
        evening_end = 23 * 3600
        light[evening_start:evening_end] = np.random.normal(60, 5, evening_end - evening_start)

        # Aggregate = sum of all appliances + base load
        base_load = np.random.normal(50, 5, len(timestamps))
        aggregate = fridge + microwave + light + base_load

        # Add some noise
        aggregate += np.random.normal(0, 10, len(timestamps))
        fridge = np.maximum(0, fridge)
        microwave = np.maximum(0, microwave)
        light = np.maximum(0, light)
        aggregate = np.maximum(0, aggregate)

        # Store as DataFrames
        self.mains = pd.DataFrame({
            'power': aggregate
        }, index=timestamps)

        self.appliances = {
            'fridge': pd.DataFrame({'power': fridge}, index=timestamps),
            'microwave': pd.DataFrame({'power': microwave}, index=timestamps),
            'light': pd.DataFrame({'power': light}, index=timestamps)
        }

        print(f"Generated sample data for {len(self.appliances)} appliances")
        print(f"Data duration: {len(timestamps)} seconds (~24 hours)")

    def load_building(self, building_id: int = 1) -> Dict:
        """
        Load a building from REDD dataset

        Args:
            building_id: Building number to load

        Returns:
            Dictionary with mains and appliance data
        """
        # For this demo, we'll generate synthetic data
        if self.mains is None:
            self.download_sample_data()

        return {
            'mains': self.mains,
            'appliances': self.appliances
        }

    def get_train_test_split(self, train_ratio: float = 0.7) -> Tuple[Dict, Dict]:
        """
        Split data into training and testing sets

        Args:
            train_ratio: Ratio of data to use for training

        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        if self.mains is None:
            self.download_sample_data()

        split_idx = int(len(self.mains) * train_ratio)

        train_data = {
            'mains': self.mains.iloc[:split_idx],
            'appliances': {name: df.iloc[:split_idx]
                          for name, df in self.appliances.items()}
        }

        test_data = {
            'mains': self.mains.iloc[split_idx:],
            'appliances': {name: df.iloc[split_idx:]
                          for name, df in self.appliances.items()}
        }

        return train_data, test_data

    def get_appliance_stats(self, appliance_name: str) -> Dict:
        """
        Get statistics for a specific appliance

        Args:
            appliance_name: Name of the appliance

        Returns:
            Dictionary with mean, std, max power consumption
        """
        if appliance_name not in self.appliances:
            raise ValueError(f"Appliance {appliance_name} not found")

        data = self.appliances[appliance_name]['power']

        return {
            'mean': data.mean(),
            'std': data.std(),
            'max': data.max(),
            'min': data.min(),
            'on_power': data[data > data.mean()].mean() if len(data[data > data.mean()]) > 0 else 0
        }
