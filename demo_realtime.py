#!/usr/bin/env python3
"""
Real-time NILM Disaggregation Demo

This script demonstrates real-time energy disaggregation using
trained NILM algorithms on REDD-like data.
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

sys.path.insert(0, 'src')

from datasets.redd_loader import REDDLoader
from algorithms.combinatorial_optimization import CombinatorialOptimization
from algorithms.fhmm import FHMM
from utils.visualization import plot_realtime_comparison
from utils.metrics import calculate_metrics, print_metrics


def simulate_realtime_disaggregation(
    algorithm,
    test_data,
    window_size=100,
    update_interval=0.1,
    total_samples=2000
):
    """
    Simulate real-time disaggregation with live plotting

    Args:
        algorithm: Trained NILM algorithm
        test_data: Test dataset
        window_size: Number of samples to display in the plot
        update_interval: Time between updates (seconds)
        total_samples: Total number of samples to process
    """
    print("\n" + "="*60)
    print("Starting Real-Time Disaggregation Simulation")
    print("="*60)

    # Enable interactive plotting
    plt.ion()

    # Data buffers
    time_buffer = deque(maxlen=window_size)
    mains_buffer = deque(maxlen=window_size)
    appliance_buffers = {
        app: deque(maxlen=window_size)
        for app in algorithm.appliance_names
    }

    mains_power = test_data['mains']['power'].values
    total_samples = min(total_samples, len(mains_power))

    print(f"Processing {total_samples} samples...")
    print(f"Window size: {window_size} samples")
    print(f"Appliances: {', '.join(algorithm.appliance_names)}")
    print("\nPress Ctrl+C to stop early\n")

    # Create initial plot
    fig = None

    try:
        for t in range(total_samples):
            # Get current aggregate power
            current_power = mains_power[t]

            # Predict appliance consumption
            # For CO: use the current sample
            # For FHMM: we'll use a simple approach
            if isinstance(algorithm, CombinatorialOptimization):
                combination = algorithm._find_best_combination(current_power)
                predictions = combination
            elif isinstance(algorithm, FHMM):
                # For real-time, we use a small window
                chunk = mains_power[max(0, t-10):t+1].reshape(-1, 1)
                pred = algorithm.disaggregate_chunk(chunk)
                predictions = {app: pred[app][-1] if len(pred[app]) > 0 else 0
                              for app in algorithm.appliance_names}
            else:
                predictions = {app: 0 for app in algorithm.appliance_names}

            # Update buffers
            time_buffer.append(t)
            mains_buffer.append(current_power)

            for app in algorithm.appliance_names:
                appliance_buffers[app].append(predictions[app])

            # Update plot every N samples
            if t % 10 == 0 or t == total_samples - 1:
                appliance_predictions = {
                    app: list(appliance_buffers[app])
                    for app in algorithm.appliance_names
                }

                fig = plot_realtime_comparison(
                    list(time_buffer),
                    list(mains_buffer),
                    appliance_predictions,
                    algorithm.appliance_names
                )

                plt.pause(update_interval)

                # Print status
                if t % 100 == 0:
                    print(f"Processed {t}/{total_samples} samples "
                          f"({100*t/total_samples:.1f}%)")

        print(f"\nCompleted processing {total_samples} samples!")
        print("\nClose the plot window to continue...")

        # Keep plot open
        plt.ioff()
        plt.show()

    except KeyboardInterrupt:
        print("\n\nStopped by user")
        plt.ioff()
        plt.show()


def main():
    """Main demo function"""
    print("\n" + "="*60)
    print("NILM Real-Time Disaggregation Demo")
    print("="*60)

    # Step 1: Load data
    print("\n[1/5] Loading REDD dataset...")
    loader = REDDLoader()
    train_data, test_data = loader.get_train_test_split(train_ratio=0.7)

    print(f"  Training samples: {len(train_data['mains'])}")
    print(f"  Testing samples: {len(test_data['mains'])}")
    print(f"  Appliances: {', '.join(train_data['appliances'].keys())}")

    # Step 2: Choose algorithm
    print("\n[2/5] Select NILM Algorithm:")
    print("  1. Combinatorial Optimization (CO)")
    print("  2. Factorial HMM (FHMM)")
    print("  3. Both (compare)")

    choice = input("\nEnter choice (1-3) [default=1]: ").strip() or "1"

    algorithms = []
    if choice == "1":
        algorithms = [("CO", CombinatorialOptimization())]
    elif choice == "2":
        algorithms = [("FHMM", FHMM(n_states=2))]
    else:
        algorithms = [
            ("CO", CombinatorialOptimization()),
            ("FHMM", FHMM(n_states=2))
        ]

    # Step 3: Train algorithms
    print("\n[3/5] Training algorithms...")
    for name, algo in algorithms:
        print(f"\nTraining {name}...")
        algo.train(train_data['mains'], train_data['appliances'])

    # Step 4: Choose demo type
    print("\n[4/5] Select demo type:")
    print("  1. Real-time simulation (live plotting)")
    print("  2. Batch disaggregation (full test set)")

    demo_choice = input("\nEnter choice (1-2) [default=1]: ").strip() or "1"

    # Step 5: Run disaggregation
    print("\n[5/5] Running disaggregation...")

    for name, algo in algorithms:
        print(f"\n{'='*60}")
        print(f"Algorithm: {name}")
        print('='*60)

        if demo_choice == "1":
            # Real-time simulation
            simulate_realtime_disaggregation(
                algo,
                test_data,
                window_size=200,
                update_interval=0.01,
                total_samples=2000
            )
        else:
            # Batch disaggregation
            num_test_samples = min(5000, len(test_data['mains']))
            predictions = algo.disaggregate(
                test_data['mains'],
                num_samples=num_test_samples
            )

            # Calculate metrics
            metrics = calculate_metrics(
                test_data['appliances'],
                predictions
            )
            print_metrics(metrics)

            # Plot results
            from utils.visualization import plot_disaggregation
            fig = plot_disaggregation(
                test_data['mains'],
                test_data['appliances'],
                predictions,
                start_idx=0,
                num_samples=1000,
                save_path=f'results_{name.lower()}.png'
            )
            plt.show()

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
