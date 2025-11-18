#!/usr/bin/env python3
"""
Simple test script for NILM algorithms

This script provides a quick way to test the NILM algorithms
without interactive prompts.
"""
import sys
sys.path.insert(0, 'src')

from datasets.redd_loader import REDDLoader
from algorithms.combinatorial_optimization import CombinatorialOptimization
from algorithms.fhmm import FHMM
from utils.visualization import plot_disaggregation, plot_power_histogram
from utils.metrics import calculate_metrics, print_metrics, calculate_energy_accuracy
import matplotlib.pyplot as plt


def main():
    """Main test function"""
    print("\n" + "="*70)
    print("Testing NILM Algorithms")
    print("="*70)

    # Load data
    print("\n[1/4] Loading dataset...")
    loader = REDDLoader()
    train_data, test_data = loader.get_train_test_split(train_ratio=0.7)

    print(f"  Training samples: {len(train_data['mains']):,}")
    print(f"  Testing samples: {len(test_data['mains']):,}")
    print(f"  Appliances: {', '.join(train_data['appliances'].keys())}")

    # Print appliance statistics
    print("\n  Appliance Statistics:")
    for app_name in train_data['appliances'].keys():
        stats = loader.get_appliance_stats(app_name)
        print(f"    {app_name.capitalize()}: "
              f"Mean={stats['mean']:.1f}W, "
              f"Max={stats['max']:.1f}W, "
              f"ON={stats['on_power']:.1f}W")

    # Visualize power distributions
    print("\n[2/4] Visualizing power distributions...")
    plot_power_histogram(train_data['appliances'], save_path='power_distributions.png')
    plt.close()

    # Test Combinatorial Optimization
    print("\n[3/4] Testing Combinatorial Optimization...")
    print("-" * 70)
    co = CombinatorialOptimization(threshold=15.0)
    co.train(train_data['mains'], train_data['appliances'])

    # Disaggregate test set (use subset for speed)
    num_test_samples = min(10000, len(test_data['mains']))
    print(f"\nDisaggregating {num_test_samples:,} test samples...")

    co_predictions = co.disaggregate(test_data['mains'], num_samples=num_test_samples)

    # Calculate metrics
    print("\nCombinatorial Optimization Results:")
    co_metrics = calculate_metrics(test_data['appliances'], co_predictions)
    print_metrics(co_metrics)

    # Energy accuracy
    co_energy = calculate_energy_accuracy(test_data['appliances'], co_predictions)
    print("Energy Accuracy:")
    for app, acc in co_energy.items():
        print(f"  {app.capitalize()}: {acc:.1f}%")

    # Plot CO results
    print("\nGenerating visualization...")
    plot_disaggregation(
        test_data['mains'],
        test_data['appliances'],
        co_predictions,
        start_idx=1000,
        num_samples=1000,
        save_path='results_co.png'
    )
    plt.close()
    print("  Saved: results_co.png")

    # Test FHMM
    print("\n[4/4] Testing Factorial HMM...")
    print("-" * 70)
    fhmm = FHMM(n_states=2)
    fhmm.train(train_data['mains'], train_data['appliances'])

    print(f"\nDisaggregating {num_test_samples:,} test samples...")
    fhmm_predictions = fhmm.disaggregate(test_data['mains'], num_samples=num_test_samples)

    # Calculate metrics
    print("\nFactorial HMM Results:")
    fhmm_metrics = calculate_metrics(test_data['appliances'], fhmm_predictions)
    print_metrics(fhmm_metrics)

    # Energy accuracy
    fhmm_energy = calculate_energy_accuracy(test_data['appliances'], fhmm_predictions)
    print("Energy Accuracy:")
    for app, acc in fhmm_energy.items():
        print(f"  {app.capitalize()}: {acc:.1f}%")

    # Plot FHMM results
    print("\nGenerating visualization...")
    plot_disaggregation(
        test_data['mains'],
        test_data['appliances'],
        fhmm_predictions,
        start_idx=1000,
        num_samples=1000,
        save_path='results_fhmm.png'
    )
    plt.close()
    print("  Saved: results_fhmm.png")

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("\nGenerated files:")
    print("  - power_distributions.png (appliance power histograms)")
    print("  - results_co.png (CO disaggregation results)")
    print("  - results_fhmm.png (FHMM disaggregation results)")

    print("\nAlgorithm Performance Comparison:")
    print(f"{'Appliance':<15} {'CO F1':<10} {'FHMM F1':<10} {'Winner':<10}")
    print("-" * 70)
    for app in co_metrics.keys():
        co_f1 = co_metrics[app]['F1']
        fhmm_f1 = fhmm_metrics[app]['F1']
        winner = 'CO' if co_f1 > fhmm_f1 else 'FHMM' if fhmm_f1 > co_f1 else 'Tie'
        print(f"{app.capitalize():<15} {co_f1:<10.3f} {fhmm_f1:<10.3f} {winner:<10}")

    print("\n" + "="*70)
    print("Testing Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
