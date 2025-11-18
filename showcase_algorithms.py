#!/usr/bin/env python3
"""
Algorithm Showcase - Demonstrate all available NILM algorithms

This script showcases all implemented NILM algorithms including:
- Baseline algorithms (Mean, Zero, Median)
- Traditional ML (kNN, CO, FHMM)
- Deep Learning (Seq2Seq, DAE) if PyTorch is available
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')

from datasets.redd_loader import REDDLoader
from datasets.real_datasets import DatasetRegistry
from algorithms import (
    CombinatorialOptimization,
    FHMM,
    MeanAlgorithm,
    ZeroAlgorithm,
    MedianAlgorithm,
    KNNDisaggregator
)
from utils.metrics import calculate_metrics, print_metrics
from utils.visualization import plot_disaggregation

# Check for deep learning
try:
    from algorithms import Seq2SeqDisaggregator, DAEDisaggregator
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def print_section(text):
    """Print a section header"""
    print("\n" + "-"*80)
    print(text)
    print("-"*80)


def main():
    """Main showcase function"""
    print_header("NILM ALGORITHM SHOWCASE")

    print("This script demonstrates all available NILM algorithms:")
    print("\n📊 Baseline Algorithms:")
    print("  • Mean Algorithm - Always predicts mean power")
    print("  • Zero Algorithm - Always predicts zero (lower bound)")
    print("  • Median Algorithm - Always predicts median power")

    print("\n🔧 Traditional ML Algorithms:")
    print("  • Combinatorial Optimization (CO) - Fast state-based approach")
    print("  • Factorial HMM (FHMM) - Probabilistic temporal model")
    print("  • k-Nearest Neighbors (kNN) - Pattern matching approach")

    if DEEP_LEARNING_AVAILABLE:
        print("\n🧠 Deep Learning Algorithms (PyTorch detected!):")
        print("  • Seq2Seq LSTM - Sequence-to-sequence encoder-decoder")
        print("  • Denoising Autoencoder (DAE) - Autoencoder-based approach")
    else:
        print("\n⚠️  Deep Learning algorithms unavailable")
        print("    Install PyTorch: uv pip install torch")

    # Check dataset availability
    print("\n" + "="*80)
    print("Dataset Availability Check")
    print("="*80)

    dataset_status = DatasetRegistry.check_availability()
    for name, available in dataset_status.items():
        status = "✓ Available" if available else "✗ Not found"
        print(f"  {name.upper()}: {status}")

    print("\n💡 Using synthetic REDD-like data for this demo")
    print("   For real datasets, download REDD or UK-DALE")

    # Load data
    print_section("Loading Data")

    loader = REDDLoader()
    train_data, test_data = loader.get_train_test_split(train_ratio=0.7)

    print(f"Training samples: {len(train_data['mains']):,}")
    print(f"Testing samples: {len(test_data['mains']):,}")
    print(f"Appliances: {', '.join(train_data['appliances'].keys())}")

    # Configure algorithms to test
    num_test_samples = 5000

    algorithms = [
        ("Mean", MeanAlgorithm()),
        ("Zero", ZeroAlgorithm()),
        ("Median", MedianAlgorithm()),
        ("CO", CombinatorialOptimization(threshold=15.0)),
        ("FHMM", FHMM(n_states=2)),
        ("kNN", KNNDisaggregator(k=5, window_size=10)),
    ]

    # Add deep learning if available
    if DEEP_LEARNING_AVAILABLE:
        print("\n⚡ Including deep learning algorithms")
        print("   Note: These will take longer to train")

        algorithms.extend([
            ("Seq2Seq", Seq2SeqDisaggregator(
                sequence_length=50,
                hidden_size=64,
                epochs=5,  # Reduced for demo
                batch_size=32
            )),
            ("DAE", DAEDisaggregator(
                window_size=50,
                encoding_dim=32,
                epochs=5,  # Reduced for demo
                batch_size=64
            ))
        ])

    # Results storage
    results = {}

    # Run each algorithm
    for name, algorithm in algorithms:
        print_header(f"Running: {name}")

        try:
            # Train
            print("Training...")
            train_start = time.time()
            algorithm.train(train_data['mains'], train_data['appliances'])
            train_time = time.time() - train_start

            # Disaggregate
            print("\nDisaggregating...")
            disagg_start = time.time()
            predictions = algorithm.disaggregate(
                test_data['mains'],
                num_samples=num_test_samples
            )
            disagg_time = time.time() - disagg_start

            # Calculate metrics
            print("\nEvaluating...")
            metrics = calculate_metrics(test_data['appliances'], predictions)

            # Store results
            results[name] = {
                'predictions': predictions,
                'metrics': metrics,
                'train_time': train_time,
                'disagg_time': disagg_time,
                'throughput': num_test_samples / disagg_time
            }

            # Print summary
            print(f"\n✓ {name} completed!")
            print(f"  Train time: {train_time:.2f}s")
            print(f"  Disagg time: {disagg_time:.2f}s")
            print(f"  Throughput: {num_test_samples/disagg_time:.0f} samples/sec")

            # Print metrics
            print_metrics(metrics)

        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()

    # Generate comparison
    print_header("Algorithm Comparison Summary")

    print(f"{'Algorithm':<15} {'Train (s)':<12} {'Disagg (s)':<12} {'Avg F1':<12} {'Throughput':<15}")
    print("="*80)

    for name, result in results.items():
        avg_f1 = np.mean([m['F1'] for m in result['metrics'].values()])

        print(
            f"{name:<15} "
            f"{result['train_time']:<12.2f} "
            f"{result['disagg_time']:<12.2f} "
            f"{avg_f1:<12.3f} "
            f"{result['throughput']:<15.0f}"
        )

    # Find winners
    print("\n" + "-"*80)
    print("Winners:")
    print("-"*80)

    fastest_train = min(results.items(), key=lambda x: x[1]['train_time'])
    fastest_disagg = min(results.items(), key=lambda x: x[1]['disagg_time'])
    most_accurate = max(results.items(), key=lambda x: np.mean([m['F1'] for m in x[1]['metrics'].values()]))

    print(f"⚡ Fastest Training: {fastest_train[0]} ({fastest_train[1]['train_time']:.2f}s)")
    print(f"⚡ Fastest Disaggregation: {fastest_disagg[0]} ({fastest_disagg[1]['disagg_time']:.2f}s)")
    print(f"🎯 Most Accurate: {most_accurate[0]} (F1: {np.mean([m['F1'] for m in most_accurate[1]['metrics'].values()]):.3f})")

    # Create visualization comparing top 3
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)

    # Get top 3 by F1 score
    sorted_results = sorted(
        results.items(),
        key=lambda x: np.mean([m['F1'] for m in x[1]['metrics'].values()]),
        reverse=True
    )

    top_3 = sorted_results[:min(3, len(sorted_results))]

    fig, axes = plt.subplots(len(top_3), 1, figsize=(15, 4 * len(top_3)))

    if len(top_3) == 1:
        axes = [axes]

    for idx, (name, result) in enumerate(top_3):
        # Plot first appliance
        app_name = list(result['predictions'].keys())[0]

        start_idx = 1000
        plot_samples = 500

        gt = test_data['appliances'][app_name]['power'].iloc[start_idx:start_idx+plot_samples].values
        pred = result['predictions'][app_name]['power'].iloc[start_idx:start_idx+plot_samples].values

        axes[idx].plot(gt, 'g-', linewidth=2, label='Ground Truth', alpha=0.7)
        axes[idx].plot(pred, 'r--', linewidth=1.5, label='Prediction', alpha=0.7)
        axes[idx].set_ylabel('Power (W)', fontsize=10)
        axes[idx].set_title(
            f'{name} - {app_name.capitalize()} (F1: {result["metrics"][app_name]["F1"]:.3f})',
            fontsize=12,
            fontweight='bold'
        )
        axes[idx].legend(loc='upper right')
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (samples)', fontsize=10)
    plt.tight_layout()
    plt.savefig('algorithm_showcase.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization: algorithm_showcase.png")

    # Create performance comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = list(results.keys())
    f1_scores = [np.mean([m['F1'] for m in results[n]['metrics'].values()]) for n in names]
    throughputs = [results[n]['throughput'] for n in names]

    # F1 scores
    bars1 = ax1.barh(names, f1_scores, color='steelblue')
    ax1.set_xlabel('Average F1 Score', fontsize=11)
    ax1.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, score in zip(bars1, f1_scores):
        ax1.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height()/2,
            f'{score:.3f}',
            va='center',
            fontsize=9
        )

    # Throughput
    bars2 = ax2.barh(names, throughputs, color='coral')
    ax2.set_xlabel('Throughput (samples/sec)', fontsize=11)
    ax2.set_title('Speed Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, tput in zip(bars2, throughputs):
        ax2.text(
            bar.get_width() + max(throughputs) * 0.02,
            bar.get_y() + bar.get_height()/2,
            f'{int(tput)}',
            va='center',
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved comparison: algorithm_comparison.png")

    # Final summary
    print_header("Showcase Complete!")

    print("Summary:")
    print(f"  • Tested {len(results)} algorithms")
    print(f"  • Processed {num_test_samples:,} test samples")
    print(f"  • Generated 2 visualization files")

    print("\nGenerated files:")
    print("  • algorithm_showcase.png - Top 3 predictions")
    print("  • algorithm_comparison.png - Performance comparison")

    print("\nNext steps:")
    print("  • Try: python benchmark.py (comprehensive benchmark)")
    print("  • Try: python demo_realtime.py (interactive demo)")
    print("  • Download real datasets (REDD, UK-DALE) for better testing")

    if DEEP_LEARNING_AVAILABLE:
        print("\nDeep Learning Tips:")
        print("  • Increase epochs for better accuracy (currently 5)")
        print("  • Try larger hidden_size for more capacity")
        print("  • Use GPU for faster training (detected: cuda available)" if __import__('torch').cuda.is_available() else "  • Install CUDA for GPU acceleration")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
