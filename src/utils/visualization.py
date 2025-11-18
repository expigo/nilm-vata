"""
Visualization utilities for NILM results
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def plot_disaggregation(
    mains: pd.DataFrame,
    ground_truth: Dict[str, pd.DataFrame],
    predictions: Dict[str, pd.DataFrame],
    start_idx: int = 0,
    num_samples: int = 1000,
    save_path: Optional[str] = None
):
    """
    Plot disaggregation results comparing ground truth and predictions

    Args:
        mains: Aggregate power consumption
        ground_truth: Dictionary of actual appliance consumption
        predictions: Dictionary of predicted appliance consumption
        start_idx: Starting index for plotting
        num_samples: Number of samples to plot
        save_path: Path to save the figure (optional)
    """
    appliances = list(predictions.keys())
    n_appliances = len(appliances)

    # Create subplots: aggregate + one per appliance
    fig, axes = plt.subplots(n_appliances + 1, 1, figsize=(15, 3 * (n_appliances + 1)))

    if n_appliances == 0:
        axes = [axes]

    end_idx = start_idx + num_samples

    # Plot aggregate power
    time_range = range(start_idx, min(end_idx, len(mains)))
    axes[0].plot(time_range, mains['power'].iloc[start_idx:end_idx], 'b-', linewidth=1.5, label='Aggregate')
    axes[0].set_ylabel('Power (W)', fontsize=10)
    axes[0].set_title('Aggregate Power Consumption', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot each appliance
    for idx, app_name in enumerate(appliances, 1):
        ax = axes[idx]

        # Ground truth
        gt_data = ground_truth[app_name]['power'].iloc[start_idx:end_idx]
        ax.plot(time_range, gt_data, 'g-', linewidth=2, label='Ground Truth', alpha=0.7)

        # Prediction
        pred_data = predictions[app_name]['power'].iloc[start_idx:end_idx]
        ax.plot(time_range, pred_data, 'r--', linewidth=1.5, label='Prediction', alpha=0.7)

        ax.set_ylabel('Power (W)', fontsize=10)
        ax.set_title(f'{app_name.capitalize()}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (samples)', fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    return fig


def plot_realtime_comparison(
    time_window: List,
    mains_data: List,
    appliance_predictions: Dict[str, List],
    appliance_names: List[str]
):
    """
    Create a real-time visualization for live disaggregation

    Args:
        time_window: List of time points
        mains_data: List of aggregate power values
        appliance_predictions: Dictionary of predicted power for each appliance
        appliance_names: List of appliance names to plot
    """
    n_plots = len(appliance_names) + 1

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2.5 * n_plots))

    if n_plots == 1:
        axes = [axes]

    # Plot aggregate
    axes[0].clear()
    axes[0].plot(time_window, mains_data, 'b-', linewidth=2)
    axes[0].set_ylabel('Power (W)', fontsize=10)
    axes[0].set_title('Aggregate Power', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot appliances
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for idx, app_name in enumerate(appliance_names, 1):
        axes[idx].clear()
        color = colors[idx % len(colors)]

        if app_name in appliance_predictions:
            axes[idx].plot(
                time_window,
                appliance_predictions[app_name],
                color=color,
                linewidth=2
            )

        axes[idx].set_ylabel('Power (W)', fontsize=10)
        axes[idx].set_title(f'{app_name.capitalize()}', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time', fontsize=10)
    plt.tight_layout()

    return fig


def plot_power_histogram(appliances: Dict[str, pd.DataFrame], save_path: Optional[str] = None):
    """
    Plot histogram of power consumption for each appliance

    Args:
        appliances: Dictionary of appliance power data
        save_path: Path to save the figure (optional)
    """
    n_appliances = len(appliances)
    fig, axes = plt.subplots(1, n_appliances, figsize=(5 * n_appliances, 4))

    if n_appliances == 1:
        axes = [axes]

    for idx, (app_name, app_data) in enumerate(appliances.items()):
        power = app_data['power'].values
        axes[idx].hist(power, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel('Power (W)', fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].set_title(f'{app_name.capitalize()} Power Distribution', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')

        # Add statistics
        mean_power = power.mean()
        axes[idx].axvline(mean_power, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_power:.1f}W')
        axes[idx].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")

    return fig
