#!/usr/bin/env python3
"""
Comprehensive benchmark tool for comparing NILM algorithms

This script runs multiple NILM algorithms and compares their performance
across various metrics and generates detailed comparison reports.
"""
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import seaborn as sns

sys.path.insert(0, 'src')

from datasets.redd_loader import REDDLoader
from algorithms.combinatorial_optimization import CombinatorialOptimization
from algorithms.fhmm import FHMM
from utils.metrics import calculate_metrics, print_metrics
from utils.model_persistence import ModelManager
from utils.energy_cost import EnergyCostCalculator


class BenchmarkRunner:
    """Run and compare multiple NILM algorithms"""

    def __init__(self, test_samples: int = 5000):
        """
        Initialize benchmark runner

        Args:
            test_samples: Number of samples to use for testing
        """
        self.test_samples = test_samples
        self.results = {}
        self.models = {}

    def prepare_data(self):
        """Prepare dataset for benchmarking"""
        print("\n" + "="*70)
        print("Preparing Dataset")
        print("="*70)

        loader = REDDLoader()
        self.train_data, self.test_data = loader.get_train_test_split(train_ratio=0.7)

        print(f"Training samples: {len(self.train_data['mains']):,}")
        print(f"Testing samples: {len(self.test_data['mains']):,}")
        print(f"Using {self.test_samples:,} samples for benchmark")
        print(f"Appliances: {', '.join(self.train_data['appliances'].keys())}")

    def run_algorithm(
        self,
        name: str,
        algorithm,
        train_params: Dict = None,
        disagg_params: Dict = None
    ):
        """
        Run a single algorithm and collect results

        Args:
            name: Algorithm name
            algorithm: Algorithm instance
            train_params: Training parameters
            disagg_params: Disaggregation parameters
        """
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print('='*70)

        train_params = train_params or {}
        disagg_params = disagg_params or {}

        # Training
        print("\nTraining...")
        train_start = time.time()
        algorithm.train(self.train_data['mains'], self.train_data['appliances'], **train_params)
        train_time = time.time() - train_start

        # Disaggregation
        print("\nDisaggregating...")
        disagg_start = time.time()
        predictions = algorithm.disaggregate(
            self.test_data['mains'],
            num_samples=self.test_samples,
            **disagg_params
        )
        disagg_time = time.time() - disagg_start

        # Calculate metrics
        print("\nCalculating metrics...")
        metrics = calculate_metrics(self.test_data['appliances'], predictions)

        # Store results
        self.results[name] = {
            'predictions': predictions,
            'metrics': metrics,
            'train_time': train_time,
            'disagg_time': disagg_time,
            'samples_per_second': self.test_samples / disagg_time
        }

        self.models[name] = algorithm

        print(f"\nTraining time: {train_time:.2f}s")
        print(f"Disaggregation time: {disagg_time:.2f}s")
        print(f"Throughput: {self.test_samples/disagg_time:.0f} samples/sec")
        print_metrics(metrics)

    def generate_comparison_report(self) -> str:
        """Generate comprehensive comparison report"""
        report = []
        report.append("\n" + "="*80)
        report.append("ALGORITHM BENCHMARK COMPARISON REPORT")
        report.append("="*80)

        # Performance summary
        report.append("\n" + "-"*80)
        report.append("PERFORMANCE SUMMARY")
        report.append("-"*80)
        report.append(f"{'Algorithm':<20} {'Train (s)':<12} {'Disagg (s)':<12} {'Throughput':<15}")
        report.append("-"*80)

        for name, result in self.results.items():
            report.append(
                f"{name:<20} "
                f"{result['train_time']:<12.2f} "
                f"{result['disagg_time']:<12.2f} "
                f"{result['samples_per_second']:<15.0f}"
            )

        # Accuracy comparison
        report.append("\n" + "-"*80)
        report.append("ACCURACY COMPARISON (Average across appliances)")
        report.append("-"*80)
        report.append(f"{'Algorithm':<20} {'MAE':<12} {'RMSE':<12} {'F1':<12} {'NDE':<12}")
        report.append("-"*80)

        for name, result in self.results.items():
            metrics = result['metrics']
            avg_mae = np.mean([m['MAE'] for m in metrics.values()])
            avg_rmse = np.mean([m['RMSE'] for m in metrics.values()])
            avg_f1 = np.mean([m['F1'] for m in metrics.values()])
            avg_nde = np.mean([m['NDE'] for m in metrics.values()])

            report.append(
                f"{name:<20} "
                f"{avg_mae:<12.2f} "
                f"{avg_rmse:<12.2f} "
                f"{avg_f1:<12.3f} "
                f"{avg_nde:<12.3f}"
            )

        # Per-appliance winners
        report.append("\n" + "-"*80)
        report.append("BEST ALGORITHM PER APPLIANCE (by F1 score)")
        report.append("-"*80)

        appliances = list(next(iter(self.results.values()))['metrics'].keys())

        for app in appliances:
            f1_scores = {
                name: result['metrics'][app]['F1']
                for name, result in self.results.items()
            }

            winner = max(f1_scores.items(), key=lambda x: x[1])
            report.append(f"{app.capitalize():<15} → {winner[0]} (F1: {winner[1]:.3f})")

        # Overall winner
        report.append("\n" + "-"*80)
        report.append("OVERALL WINNER")
        report.append("-"*80)

        overall_scores = {}
        for name, result in self.results.items():
            # Composite score: lower is better for MAE, RMSE, NDE; higher for F1
            metrics = result['metrics']
            avg_f1 = np.mean([m['F1'] for m in metrics.values()])
            avg_nde = np.mean([m['NDE'] for m in metrics.values()])

            # Simple scoring: F1 (maximize) - NDE (minimize)
            overall_scores[name] = avg_f1 - avg_nde

        winner = max(overall_scores.items(), key=lambda x: x[1])
        report.append(f"Winner: {winner[0]} (Score: {winner[1]:.3f})")

        # Recommendations
        report.append("\n" + "-"*80)
        report.append("RECOMMENDATIONS")
        report.append("-"*80)

        fastest = min(self.results.items(), key=lambda x: x[1]['disagg_time'])
        most_accurate = max(self.results.items(), key=lambda x: np.mean([m['F1'] for m in x[1]['metrics'].values()]))

        report.append(f"• For SPEED: Use {fastest[0]} ({fastest[1]['samples_per_second']:.0f} samples/sec)")
        report.append(f"• For ACCURACY: Use {most_accurate[0]} (F1: {np.mean([m['F1'] for m in most_accurate[1]['metrics'].values()]):.3f})")

        report.append("="*80 + "\n")

        return "\n".join(report)

    def plot_comparison(self, save_path: str = 'benchmark_comparison.png'):
        """Create visualization comparing algorithms"""
        n_algorithms = len(self.results)
        appliances = list(next(iter(self.results.values()))['metrics'].keys())

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. F1 Score comparison
        ax1 = fig.add_subplot(gs[0, 0])
        data = []
        for name, result in self.results.items():
            for app in appliances:
                data.append({
                    'Algorithm': name,
                    'Appliance': app.capitalize(),
                    'F1 Score': result['metrics'][app]['F1']
                })

        df = pd.DataFrame(data)
        df_pivot = df.pivot(index='Appliance', columns='Algorithm', values='F1 Score')
        df_pivot.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('F1 Score Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylabel('F1 Score', fontsize=10)
        ax1.legend(title='Algorithm', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1)

        # 2. MAE comparison
        ax2 = fig.add_subplot(gs[0, 1])
        data = []
        for name, result in self.results.items():
            for app in appliances:
                data.append({
                    'Algorithm': name,
                    'Appliance': app.capitalize(),
                    'MAE': result['metrics'][app]['MAE']
                })

        df = pd.DataFrame(data)
        df_pivot = df.pivot(index='Appliance', columns='Algorithm', values='MAE')
        df_pivot.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('MAE (Watts)', fontsize=10)
        ax2.legend(title='Algorithm', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Speed comparison
        ax3 = fig.add_subplot(gs[1, 0])
        names = list(self.results.keys())
        train_times = [self.results[n]['train_time'] for n in names]
        disagg_times = [self.results[n]['disagg_time'] for n in names]

        x = np.arange(len(names))
        width = 0.35

        ax3.bar(x - width/2, train_times, width, label='Training', color='steelblue')
        ax3.bar(x + width/2, disagg_times, width, label='Disaggregation', color='coral')
        ax3.set_ylabel('Time (seconds)', fontsize=10)
        ax3.set_title('Execution Time Comparison', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(names)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Throughput comparison
        ax4 = fig.add_subplot(gs[1, 1])
        throughputs = [self.results[n]['samples_per_second'] for n in names]
        bars = ax4.bar(names, throughputs, color='green', alpha=0.7)
        ax4.set_ylabel('Samples/Second', fontsize=10)
        ax4.set_title('Throughput Comparison', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9
            )

        # 5. Overall metrics heatmap
        ax5 = fig.add_subplot(gs[2, :])

        metrics_data = []
        for name in names:
            result = self.results[name]
            metrics = result['metrics']

            row = [
                np.mean([m['MAE'] for m in metrics.values()]),
                np.mean([m['RMSE'] for m in metrics.values()]),
                np.mean([m['F1'] for m in metrics.values()]),
                np.mean([m['NDE'] for m in metrics.values()]),
            ]
            metrics_data.append(row)

        # Normalize metrics for heatmap (0-1 scale)
        metrics_array = np.array(metrics_data).T
        normalized = np.zeros_like(metrics_array)

        # For MAE, RMSE, NDE: lower is better, so invert
        for i in [0, 1, 3]:
            col = metrics_array[i]
            if col.max() > 0:
                normalized[i] = 1 - (col / col.max())

        # For F1: higher is better
        col = metrics_array[2]
        if col.max() > 0:
            normalized[2] = col / col.max()

        sns.heatmap(
            normalized,
            annot=metrics_array,
            fmt='.2f',
            xticklabels=names,
            yticklabels=['MAE', 'RMSE', 'F1', 'NDE'],
            cmap='RdYlGn',
            ax=ax5,
            cbar_kws={'label': 'Normalized Score (Higher is Better)'}
        )
        ax5.set_title('Overall Performance Heatmap', fontsize=12, fontweight='bold')

        plt.suptitle(
            f'NILM Algorithm Benchmark Comparison ({self.test_samples:,} samples)',
            fontsize=14,
            fontweight='bold',
            y=0.995
        )

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to: {save_path}")

        return fig


def main():
    """Main benchmark function"""
    print("\n" + "="*70)
    print("NILM ALGORITHM BENCHMARK")
    print("="*70)

    # Configuration
    num_samples = 10000

    # Initialize benchmark
    runner = BenchmarkRunner(test_samples=num_samples)

    # Prepare data
    runner.prepare_data()

    # Run algorithms
    print("\n[1/3] Running Combinatorial Optimization...")
    co = CombinatorialOptimization(threshold=15.0)
    runner.run_algorithm("Combinatorial Optimization", co)

    print("\n[2/3] Running Factorial HMM...")
    fhmm = FHMM(n_states=2)
    runner.run_algorithm("Factorial HMM", fhmm)

    # You can add more algorithms here
    # print("\n[3/3] Running Another Algorithm...")
    # another_algo = AnotherAlgorithm()
    # runner.run_algorithm("Another Algorithm", another_algo)

    # Generate report
    print("\n[3/3] Generating comparison report...")
    report = runner.generate_comparison_report()
    print(report)

    # Save report
    with open('benchmark_report.txt', 'w') as f:
        f.write(report)
    print("Report saved to: benchmark_report.txt")

    # Generate visualizations
    runner.plot_comparison('benchmark_comparison.png')

    # Optional: Save best model
    print("\n" + "="*70)
    print("Model Persistence")
    print("="*70)

    model_manager = ModelManager()

    for name, model in runner.models.items():
        avg_f1 = np.mean([m['F1'] for m in runner.results[name]['metrics'].values()])

        metadata = {
            'test_samples': num_samples,
            'avg_f1_score': avg_f1,
            'train_time': runner.results[name]['train_time'],
            'disagg_time': runner.results[name]['disagg_time'],
        }

        model_path = model_manager.save_model(
            model,
            f"benchmark_{name.lower().replace(' ', '_')}",
            name.replace(' ', '_'),
            metadata
        )

    # Energy cost analysis
    print("\n" + "="*70)
    print("Energy Cost Analysis")
    print("="*70)

    cost_calc = EnergyCostCalculator(
        rate_per_kwh=0.12,
        peak_rate=0.18,
        peak_hours=(17, 21)
    )

    # Use a subset for cost calculation
    cost_data = {
        app: runner.test_data['appliances'][app].iloc[:runner.test_samples]
        for app in runner.test_data['appliances'].keys()
    }

    report = cost_calc.generate_cost_report(cost_data)
    print(report)

    cost_calc.plot_cost_breakdown(cost_data, save_path='energy_cost_breakdown.png')

    print("\n" + "="*70)
    print("Benchmark Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - benchmark_report.txt")
    print("  - benchmark_comparison.png")
    print("  - energy_cost_breakdown.png")
    print("  - Model files in ./models/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
