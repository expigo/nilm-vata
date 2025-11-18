"""
Baseline NILM algorithms for comparison

These simple algorithms provide baselines to compare against more sophisticated methods.
"""
import numpy as np
import pandas as pd
from typing import Dict


class MeanAlgorithm:
    """
    Mean algorithm - predicts the mean power consumption for each appliance

    This is the simplest possible algorithm. It learns the average power
    consumption of each appliance during training and always predicts that value.
    """

    def __init__(self):
        """Initialize Mean algorithm"""
        self.appliance_means = {}
        self.appliance_names = []

    def train(self, mains: pd.DataFrame, appliances: Dict[str, pd.DataFrame]):
        """
        Train by computing mean power for each appliance

        Args:
            mains: Aggregate power consumption (not used)
            appliances: Dictionary of appliance-wise power data
        """
        print("Training Mean algorithm...")
        self.appliance_names = list(appliances.keys())

        for app_name, app_data in appliances.items():
            mean_power = app_data['power'].mean()
            self.appliance_means[app_name] = mean_power
            print(f"  {app_name}: Mean power = {mean_power:.2f}W")

    def disaggregate(self, mains: pd.DataFrame, num_samples: int = None) -> Dict[str, pd.DataFrame]:
        """
        Predict by returning constant mean values

        Args:
            mains: Aggregate power consumption
            num_samples: Number of samples to process

        Returns:
            Dictionary of constant predictions
        """
        print("Disaggregating with Mean algorithm...")

        if num_samples is None:
            num_samples = len(mains)

        predictions = {}

        for app_name in self.appliance_names:
            mean_value = self.appliance_means[app_name]
            predictions[app_name] = pd.DataFrame(
                {'power': np.full(num_samples, mean_value)},
                index=mains.index[:num_samples]
            )

        print("Disaggregation complete!")
        return predictions


class ZeroAlgorithm:
    """
    Zero algorithm - always predicts zero power

    This is the worst possible algorithm, used as a lower bound for comparison.
    """

    def __init__(self):
        """Initialize Zero algorithm"""
        self.appliance_names = []

    def train(self, mains: pd.DataFrame, appliances: Dict[str, pd.DataFrame]):
        """
        Training is trivial - just remember appliance names

        Args:
            mains: Aggregate power consumption (not used)
            appliances: Dictionary of appliance-wise power data
        """
        print("Training Zero algorithm (no-op)...")
        self.appliance_names = list(appliances.keys())

    def disaggregate(self, mains: pd.DataFrame, num_samples: int = None) -> Dict[str, pd.DataFrame]:
        """
        Predict all zeros

        Args:
            mains: Aggregate power consumption
            num_samples: Number of samples to process

        Returns:
            Dictionary of zero predictions
        """
        print("Disaggregating with Zero algorithm...")

        if num_samples is None:
            num_samples = len(mains)

        predictions = {}

        for app_name in self.appliance_names:
            predictions[app_name] = pd.DataFrame(
                {'power': np.zeros(num_samples)},
                index=mains.index[:num_samples]
            )

        print("Disaggregation complete!")
        return predictions


class MedianAlgorithm:
    """
    Median algorithm - predicts the median power consumption

    Similar to Mean but uses median, which is more robust to outliers.
    """

    def __init__(self):
        """Initialize Median algorithm"""
        self.appliance_medians = {}
        self.appliance_names = []

    def train(self, mains: pd.DataFrame, appliances: Dict[str, pd.DataFrame]):
        """
        Train by computing median power for each appliance

        Args:
            mains: Aggregate power consumption (not used)
            appliances: Dictionary of appliance-wise power data
        """
        print("Training Median algorithm...")
        self.appliance_names = list(appliances.keys())

        for app_name, app_data in appliances.items():
            median_power = app_data['power'].median()
            self.appliance_medians[app_name] = median_power
            print(f"  {app_name}: Median power = {median_power:.2f}W")

    def disaggregate(self, mains: pd.DataFrame, num_samples: int = None) -> Dict[str, pd.DataFrame]:
        """
        Predict by returning constant median values

        Args:
            mains: Aggregate power consumption
            num_samples: Number of samples to process

        Returns:
            Dictionary of constant predictions
        """
        print("Disaggregating with Median algorithm...")

        if num_samples is None:
            num_samples = len(mains)

        predictions = {}

        for app_name in self.appliance_names:
            median_value = self.appliance_medians[app_name]
            predictions[app_name] = pd.DataFrame(
                {'power': np.full(num_samples, median_value)},
                index=mains.index[:num_samples]
            )

        print("Disaggregation complete!")
        return predictions
