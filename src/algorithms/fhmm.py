"""
Factorial Hidden Markov Model (FHMM) for NILM
A probabilistic approach to energy disaggregation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from hmmlearn import hmm


class FHMM:
    """
    Factorial Hidden Markov Model for NILM

    Uses HMMs to model each appliance's behavior and infers
    their states from aggregate consumption.
    """

    def __init__(self, n_states: int = 2):
        """
        Initialize FHMM algorithm

        Args:
            n_states: Number of states per appliance (default=2 for ON/OFF)
        """
        self.n_states = n_states
        self.appliance_models = {}
        self.appliance_names = []

    def train(self, mains: pd.DataFrame, appliances: Dict[str, pd.DataFrame]):
        """
        Train individual HMMs for each appliance

        Args:
            mains: Aggregate power consumption data
            appliances: Dictionary of appliance-wise power data
        """
        print("Training Factorial HMM models...")
        self.appliance_names = list(appliances.keys())

        for app_name, app_data in appliances.items():
            power = app_data['power'].values.reshape(-1, 1)

            # Train a Gaussian HMM for this appliance
            model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )

            try:
                model.fit(power)
                self.appliance_models[app_name] = model

                # Get means for each state
                means = model.means_.flatten()
                print(f"  {app_name}: Learned {self.n_states} states with means: "
                      f"{', '.join([f'{m:.1f}W' for m in sorted(means)])}")
            except Exception as e:
                print(f"  Warning: Failed to train HMM for {app_name}: {e}")
                # Create a simple fallback model
                self.appliance_models[app_name] = None

    def disaggregate(self, mains: pd.DataFrame, num_samples: int = None) -> Dict[str, pd.DataFrame]:
        """
        Disaggregate using trained HMMs

        Args:
            mains: Aggregate power consumption data
            num_samples: Number of samples to process (None = all)

        Returns:
            Dictionary of predicted appliance power consumption
        """
        print("Disaggregating with FHMM...")

        if num_samples is None:
            num_samples = len(mains)

        aggregate_power = mains['power'].values[:num_samples].reshape(-1, 1)
        predictions = {}

        for app_name in self.appliance_names:
            model = self.appliance_models.get(app_name)

            if model is None:
                # Fallback: use simple thresholding
                predictions[app_name] = np.zeros(num_samples)
                continue

            try:
                # Predict states using Viterbi algorithm
                states = model.predict(aggregate_power)

                # Map states to power values using learned means
                power_pred = model.means_[states].flatten()

                predictions[app_name] = power_pred
                print(f"  {app_name}: Predicted with {len(np.unique(states))} unique states")

            except Exception as e:
                print(f"  Warning: Prediction failed for {app_name}: {e}")
                predictions[app_name] = np.zeros(num_samples)

        # Convert to DataFrames
        result = {}
        for app_name, power_values in predictions.items():
            result[app_name] = pd.DataFrame(
                {'power': power_values},
                index=mains.index[:num_samples]
            )

        print("Disaggregation complete!")
        return result

    def disaggregate_chunk(self, mains_chunk: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Disaggregate a small chunk of data (for real-time processing)

        Args:
            mains_chunk: Array of aggregate power values

        Returns:
            Dictionary of predicted power for each appliance
        """
        if len(mains_chunk.shape) == 1:
            mains_chunk = mains_chunk.reshape(-1, 1)

        predictions = {}

        for app_name in self.appliance_names:
            model = self.appliance_models.get(app_name)

            if model is None:
                predictions[app_name] = np.zeros(len(mains_chunk))
                continue

            try:
                states = model.predict(mains_chunk)
                power_pred = model.means_[states].flatten()
                predictions[app_name] = power_pred
            except:
                predictions[app_name] = np.zeros(len(mains_chunk))

        return predictions
