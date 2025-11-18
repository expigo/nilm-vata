"""
k-Nearest Neighbors (kNN) algorithm for NILM

Uses kNN to find similar aggregate power patterns and predict appliance consumption.
"""
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.neighbors import KNeighborsRegressor


class KNNDisaggregator:
    """
    k-Nearest Neighbors based disaggregation

    For each aggregate power sample, finds k most similar samples from training
    data and uses their appliance consumption to predict.
    """

    def __init__(self, k: int = 5, window_size: int = 10):
        """
        Initialize kNN disaggregator

        Args:
            k: Number of neighbors to consider
            window_size: Size of sliding window for features
        """
        self.k = k
        self.window_size = window_size
        self.appliance_models = {}
        self.appliance_names = []

    def train(self, mains: pd.DataFrame, appliances: Dict[str, pd.DataFrame]):
        """
        Train kNN models for each appliance

        Args:
            mains: Aggregate power consumption data
            appliances: Dictionary of appliance-wise power data
        """
        print(f"Training kNN algorithm (k={self.k}, window={self.window_size})...")
        self.appliance_names = list(appliances.keys())

        # Prepare training data with sliding windows
        X_train = self._create_features(mains['power'].values)

        for app_name, app_data in appliances.items():
            y_train = app_data['power'].values

            # Align lengths
            min_len = min(len(X_train), len(y_train))
            X_train_app = X_train[:min_len]
            y_train_app = y_train[:min_len]

            # Train kNN model
            model = KNeighborsRegressor(
                n_neighbors=self.k,
                weights='distance',  # Weight by inverse distance
                metric='euclidean'
            )

            model.fit(X_train_app, y_train_app)
            self.appliance_models[app_name] = model

            print(f"  {app_name}: Trained kNN model with {len(X_train_app)} samples")

    def disaggregate(self, mains: pd.DataFrame, num_samples: int = None) -> Dict[str, pd.DataFrame]:
        """
        Disaggregate using trained kNN models

        Args:
            mains: Aggregate power consumption data
            num_samples: Number of samples to process

        Returns:
            Dictionary of predicted appliance power consumption
        """
        print("Disaggregating with kNN algorithm...")

        if num_samples is None:
            num_samples = len(mains)

        # Create features
        X_test = self._create_features(mains['power'].values[:num_samples])

        predictions = {}

        for app_name in self.appliance_names:
            model = self.appliance_models[app_name]

            # Predict
            y_pred = model.predict(X_test)

            # Ensure non-negative
            y_pred = np.maximum(0, y_pred)

            predictions[app_name] = pd.DataFrame(
                {'power': y_pred},
                index=mains.index[:len(y_pred)]
            )

            print(f"  {app_name}: Predicted {len(y_pred)} samples")

        print("Disaggregation complete!")
        return predictions

    def _create_features(self, power_data: np.ndarray) -> np.ndarray:
        """
        Create features using sliding window

        Args:
            power_data: Array of power values

        Returns:
            Feature matrix with sliding windows
        """
        n_samples = len(power_data) - self.window_size + 1

        if n_samples <= 0:
            # Not enough data for windows, use padding
            features = power_data.reshape(-1, 1)
        else:
            features = np.zeros((n_samples, self.window_size))

            for i in range(n_samples):
                features[i] = power_data[i:i + self.window_size]

        return features
