"""
Combinatorial Optimization (CO) Algorithm for NILM
A simple but effective approach for energy disaggregation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from itertools import product


class CombinatorialOptimization:
    """
    Combinatorial Optimization algorithm for NILM

    This algorithm works by finding the best combination of appliance states
    that minimizes the difference between predicted and actual aggregate power.
    """

    def __init__(self, threshold: float = 15.0):
        """
        Initialize CO algorithm

        Args:
            threshold: Power threshold to determine on/off states (watts)
        """
        self.threshold = threshold
        self.appliance_models = {}
        self.appliance_names = []

    def train(self, mains: pd.DataFrame, appliances: Dict[str, pd.DataFrame]):
        """
        Train the CO model by learning appliance power states

        Args:
            mains: Aggregate power consumption data
            appliances: Dictionary of appliance-wise power data
        """
        print("Training Combinatorial Optimization model...")
        self.appliance_names = list(appliances.keys())

        for app_name, app_data in appliances.items():
            power = app_data['power'].values

            # Determine ON state power (mean of values above threshold)
            on_power = power[power > self.threshold]

            if len(on_power) > 0:
                on_mean = np.mean(on_power)
                on_std = np.std(on_power)
            else:
                on_mean = 0
                on_std = 0

            self.appliance_models[app_name] = {
                'on_power': on_mean,
                'on_std': on_std,
                'off_power': np.mean(power[power <= self.threshold]),
                'threshold': self.threshold
            }

            print(f"  {app_name}: ON={on_mean:.1f}W (±{on_std:.1f}), "
                  f"OFF={self.appliance_models[app_name]['off_power']:.1f}W")

    def disaggregate(self, mains: pd.DataFrame, num_samples: int = None) -> Dict[str, pd.DataFrame]:
        """
        Disaggregate the aggregate power into individual appliances

        Args:
            mains: Aggregate power consumption data
            num_samples: Number of samples to process (None = all)

        Returns:
            Dictionary of predicted appliance power consumption
        """
        print("Disaggregating with Combinatorial Optimization...")

        if num_samples is None:
            num_samples = len(mains)

        aggregate_power = mains['power'].values[:num_samples]
        predictions = {app: np.zeros(num_samples) for app in self.appliance_names}

        # For each time step, find the best combination of appliance states
        for t in range(num_samples):
            if t % 10000 == 0:
                print(f"  Processing sample {t}/{num_samples}...")

            target_power = aggregate_power[t]

            # Generate all possible combinations of ON/OFF states
            # For efficiency, we limit to 3 appliances max for full search
            if len(self.appliance_names) <= 5:
                best_combination = self._find_best_combination(target_power)
            else:
                # For more appliances, use greedy approach
                best_combination = self._greedy_combination(target_power)

            # Assign predicted states
            for app_name, state in best_combination.items():
                predictions[app_name][t] = state

        # Convert to DataFrames
        result = {}
        for app_name, power_values in predictions.items():
            result[app_name] = pd.DataFrame(
                {'power': power_values},
                index=mains.index[:num_samples]
            )

        print("Disaggregation complete!")
        return result

    def _find_best_combination(self, target_power: float) -> Dict[str, float]:
        """
        Find the best combination of appliance states using exhaustive search

        Args:
            target_power: Target aggregate power

        Returns:
            Dictionary of appliance states
        """
        best_error = float('inf')
        best_combination = {app: 0 for app in self.appliance_names}

        # Generate all possible ON/OFF combinations
        states = [[model['off_power'], model['on_power']]
                 for model in [self.appliance_models[app] for app in self.appliance_names]]

        for combination in product(*states):
            predicted_power = sum(combination)
            error = abs(predicted_power - target_power)

            if error < best_error:
                best_error = error
                best_combination = {app: power for app, power in zip(self.appliance_names, combination)}

        return best_combination

    def _greedy_combination(self, target_power: float) -> Dict[str, float]:
        """
        Find a good combination using greedy approach (for many appliances)

        Args:
            target_power: Target aggregate power

        Returns:
            Dictionary of appliance states
        """
        remaining_power = target_power
        combination = {}

        # Sort appliances by ON power (descending)
        sorted_apps = sorted(self.appliance_names,
                           key=lambda x: self.appliance_models[x]['on_power'],
                           reverse=True)

        for app in sorted_apps:
            model = self.appliance_models[app]
            on_power = model['on_power']
            off_power = model['off_power']

            # Decide if turning on this appliance reduces error
            error_if_on = abs(remaining_power - on_power)
            error_if_off = abs(remaining_power - off_power)

            if error_if_on < error_if_off:
                combination[app] = on_power
                remaining_power -= on_power
            else:
                combination[app] = off_power
                remaining_power -= off_power

        return combination
