"""
Metrics for evaluating NILM algorithm performance
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(
    ground_truth: Dict[str, pd.DataFrame],
    predictions: Dict[str, pd.DataFrame]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate performance metrics for NILM predictions

    Args:
        ground_truth: Dictionary of actual appliance consumption
        predictions: Dictionary of predicted appliance consumption

    Returns:
        Dictionary of metrics for each appliance
    """
    metrics = {}

    for app_name in ground_truth.keys():
        if app_name not in predictions:
            continue

        gt = ground_truth[app_name]['power'].values
        pred = predictions[app_name]['power'].values

        # Ensure same length
        min_len = min(len(gt), len(pred))
        gt = gt[:min_len]
        pred = pred[:min_len]

        # Calculate metrics
        mae = mean_absolute_error(gt, pred)
        rmse = np.sqrt(mean_squared_error(gt, pred))

        # Normalized metrics
        gt_energy = np.sum(gt)
        pred_energy = np.sum(pred)

        if gt_energy > 0:
            nde = abs(gt_energy - pred_energy) / gt_energy  # Normalized Disaggregation Error
        else:
            nde = 0

        # F1 score for ON/OFF detection
        threshold = gt.mean() * 0.5
        gt_binary = (gt > threshold).astype(int)
        pred_binary = (pred > threshold).astype(int)

        tp = np.sum((gt_binary == 1) & (pred_binary == 1))
        fp = np.sum((gt_binary == 0) & (pred_binary == 1))
        fn = np.sum((gt_binary == 1) & (pred_binary == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[app_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'NDE': nde,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }

    return metrics


def print_metrics(metrics: Dict[str, Dict[str, float]]):
    """
    Print metrics in a formatted table

    Args:
        metrics: Dictionary of metrics for each appliance
    """
    print("\n" + "="*80)
    print(f"{'Appliance':<15} {'MAE (W)':<12} {'RMSE (W)':<12} {'NDE':<10} {'F1 Score':<10}")
    print("="*80)

    for app_name, app_metrics in metrics.items():
        print(f"{app_name:<15} "
              f"{app_metrics['MAE']:<12.2f} "
              f"{app_metrics['RMSE']:<12.2f} "
              f"{app_metrics['NDE']:<10.3f} "
              f"{app_metrics['F1']:<10.3f}")

    print("="*80)

    # Calculate average metrics
    avg_mae = np.mean([m['MAE'] for m in metrics.values()])
    avg_rmse = np.mean([m['RMSE'] for m in metrics.values()])
    avg_nde = np.mean([m['NDE'] for m in metrics.values()])
    avg_f1 = np.mean([m['F1'] for m in metrics.values()])

    print(f"{'AVERAGE':<15} "
          f"{avg_mae:<12.2f} "
          f"{avg_rmse:<12.2f} "
          f"{avg_nde:<10.3f} "
          f"{avg_f1:<10.3f}")
    print("="*80 + "\n")


def calculate_energy_accuracy(
    ground_truth: Dict[str, pd.DataFrame],
    predictions: Dict[str, pd.DataFrame]
) -> Dict[str, float]:
    """
    Calculate energy accuracy for each appliance

    Args:
        ground_truth: Dictionary of actual appliance consumption
        predictions: Dictionary of predicted appliance consumption

    Returns:
        Dictionary of energy accuracy percentages
    """
    accuracy = {}

    for app_name in ground_truth.keys():
        if app_name not in predictions:
            continue

        gt_energy = ground_truth[app_name]['power'].sum()
        pred_energy = predictions[app_name]['power'].sum()

        if gt_energy > 0:
            acc = max(0, 100 * (1 - abs(gt_energy - pred_energy) / gt_energy))
        else:
            acc = 100.0 if pred_energy == 0 else 0.0

        accuracy[app_name] = acc

    return accuracy
