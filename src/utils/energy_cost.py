"""
Energy cost calculation and analysis utilities
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import matplotlib.pyplot as plt


class EnergyCostCalculator:
    """Calculate energy costs and provide cost-saving recommendations"""

    def __init__(
        self,
        rate_per_kwh: float = 0.12,
        peak_rate: Optional[float] = None,
        peak_hours: Optional[tuple] = None
    ):
        """
        Initialize energy cost calculator

        Args:
            rate_per_kwh: Standard electricity rate ($/kWh)
            peak_rate: Peak hour rate ($/kWh), optional
            peak_hours: Tuple of (start_hour, end_hour) for peak pricing, e.g., (17, 21)
        """
        self.rate_per_kwh = rate_per_kwh
        self.peak_rate = peak_rate or rate_per_kwh
        self.peak_hours = peak_hours

    def calculate_appliance_cost(
        self,
        appliance_data: pd.DataFrame,
        time_period: str = 'daily'
    ) -> Dict[str, float]:
        """
        Calculate cost for an appliance

        Args:
            appliance_data: DataFrame with 'power' column (in watts)
            time_period: 'hourly', 'daily', 'weekly', 'monthly', 'yearly'

        Returns:
            Dictionary with cost information
        """
        # Convert to kWh (assuming 1-second sampling)
        energy_kwh = appliance_data['power'].sum() / (1000 * 3600)

        # Calculate costs based on time of use if peak hours defined
        if self.peak_hours and hasattr(appliance_data.index, 'hour'):
            peak_mask = (
                (appliance_data.index.hour >= self.peak_hours[0]) &
                (appliance_data.index.hour < self.peak_hours[1])
            )
            peak_energy = appliance_data.loc[peak_mask, 'power'].sum() / (1000 * 3600)
            off_peak_energy = appliance_data.loc[~peak_mask, 'power'].sum() / (1000 * 3600)

            cost = (peak_energy * self.peak_rate) + (off_peak_energy * self.rate_per_kwh)
        else:
            cost = energy_kwh * self.rate_per_kwh

        # Scale to requested time period
        scaling_factors = {
            'hourly': 24,
            'daily': 1,
            'weekly': 1/7,
            'monthly': 1/30,
            'yearly': 1/365
        }

        if time_period in scaling_factors:
            scale = scaling_factors[time_period]
            energy_kwh *= scale
            cost *= scale

        return {
            'energy_kwh': energy_kwh,
            'cost': cost,
            'time_period': time_period
        }

    def calculate_total_cost(
        self,
        appliances: Dict[str, pd.DataFrame],
        time_period: str = 'daily'
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate costs for all appliances

        Args:
            appliances: Dictionary of appliance DataFrames
            time_period: Time period for calculation

        Returns:
            Dictionary of cost information per appliance
        """
        costs = {}

        for app_name, app_data in appliances.items():
            costs[app_name] = self.calculate_appliance_cost(app_data, time_period)

        # Add total
        total_energy = sum(c['energy_kwh'] for c in costs.values())
        total_cost = sum(c['cost'] for c in costs.values())

        costs['TOTAL'] = {
            'energy_kwh': total_energy,
            'cost': total_cost,
            'time_period': time_period
        }

        return costs

    def generate_cost_report(
        self,
        appliances: Dict[str, pd.DataFrame],
        predictions: Optional[Dict[str, pd.DataFrame]] = None
    ) -> str:
        """
        Generate a detailed cost report

        Args:
            appliances: Ground truth appliance data
            predictions: Predicted appliance data (optional)

        Returns:
            Formatted report string
        """
        report = []
        report.append("="*70)
        report.append("ENERGY COST REPORT")
        report.append("="*70)
        report.append(f"\nElectricity Rate: ${self.rate_per_kwh:.3f}/kWh")

        if self.peak_hours:
            report.append(f"Peak Rate: ${self.peak_rate:.3f}/kWh "
                         f"({self.peak_hours[0]}:00-{self.peak_hours[1]}:00)")

        # Daily costs
        report.append("\n" + "-"*70)
        report.append("DAILY COSTS")
        report.append("-"*70)

        daily_costs = self.calculate_total_cost(appliances, 'daily')

        report.append(f"{'Appliance':<15} {'Energy (kWh)':<15} {'Cost ($)':<15} {'%':<10}")
        report.append("-"*70)

        total_cost = daily_costs['TOTAL']['cost']

        for app_name in sorted(daily_costs.keys()):
            if app_name == 'TOTAL':
                continue

            info = daily_costs[app_name]
            percentage = (info['cost'] / total_cost * 100) if total_cost > 0 else 0

            report.append(
                f"{app_name:<15} "
                f"{info['energy_kwh']:<15.3f} "
                f"{info['cost']:<15.2f} "
                f"{percentage:<10.1f}"
            )

        report.append("-"*70)
        report.append(
            f"{'TOTAL':<15} "
            f"{daily_costs['TOTAL']['energy_kwh']:<15.3f} "
            f"{daily_costs['TOTAL']['cost']:<15.2f} "
            f"{'100.0':<10}"
        )

        # Monthly and yearly projections
        monthly_costs = self.calculate_total_cost(appliances, 'monthly')
        yearly_costs = self.calculate_total_cost(appliances, 'yearly')

        report.append("\n" + "-"*70)
        report.append("PROJECTED COSTS")
        report.append("-"*70)
        report.append(f"Monthly:  ${monthly_costs['TOTAL']['cost']:.2f} "
                     f"({monthly_costs['TOTAL']['energy_kwh']:.1f} kWh)")
        report.append(f"Yearly:   ${yearly_costs['TOTAL']['cost']:.2f} "
                     f"({yearly_costs['TOTAL']['energy_kwh']:.1f} kWh)")

        # Recommendations
        report.append("\n" + "-"*70)
        report.append("COST-SAVING RECOMMENDATIONS")
        report.append("-"*70)

        recommendations = self._generate_recommendations(appliances, daily_costs)
        for rec in recommendations:
            report.append(f"• {rec}")

        if predictions:
            report.append("\n" + "-"*70)
            report.append("PREDICTION ACCURACY")
            report.append("-"*70)

            pred_costs = self.calculate_total_cost(predictions, 'daily')

            for app_name in appliances.keys():
                if app_name in pred_costs:
                    actual = daily_costs[app_name]['cost']
                    predicted = pred_costs[app_name]['cost']
                    error = abs(actual - predicted) / actual * 100 if actual > 0 else 0

                    report.append(
                        f"{app_name}: "
                        f"Actual=${actual:.2f}, "
                        f"Predicted=${predicted:.2f}, "
                        f"Error={error:.1f}%"
                    )

        report.append("="*70 + "\n")

        return "\n".join(report)

    def _generate_recommendations(
        self,
        appliances: Dict[str, pd.DataFrame],
        costs: Dict[str, Dict[str, float]]
    ) -> list:
        """Generate cost-saving recommendations based on usage patterns"""
        recommendations = []

        # Find highest cost appliances
        sorted_costs = sorted(
            [(k, v['cost']) for k, v in costs.items() if k != 'TOTAL'],
            key=lambda x: x[1],
            reverse=True
        )

        if sorted_costs:
            highest_cost_app, highest_cost = sorted_costs[0]
            recommendations.append(
                f"{highest_cost_app.capitalize()} is your highest cost appliance "
                f"(${highest_cost:.2f}/day). Consider reducing usage."
            )

        # Check for always-on appliances
        for app_name, app_data in appliances.items():
            avg_power = app_data['power'].mean()
            if avg_power > 10:  # More than 10W average suggests always-on
                daily_cost = costs[app_name]['cost']
                recommendations.append(
                    f"{app_name.capitalize()} appears to be always-on "
                    f"(${daily_cost:.2f}/day). Consider using power strips or timers."
                )

        # Peak hour usage
        if self.peak_hours:
            recommendations.append(
                f"Shift high-power appliance usage outside peak hours "
                f"({self.peak_hours[0]}:00-{self.peak_hours[1]}:00) to save on costs."
            )

        return recommendations

    def plot_cost_breakdown(
        self,
        appliances: Dict[str, pd.DataFrame],
        save_path: Optional[str] = None
    ):
        """
        Create a pie chart showing cost breakdown by appliance

        Args:
            appliances: Dictionary of appliance data
            save_path: Path to save the plot
        """
        costs = self.calculate_total_cost(appliances, 'daily')

        # Remove TOTAL from plot
        plot_data = {k: v['cost'] for k, v in costs.items() if k != 'TOTAL'}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart
        colors = plt.cm.Set3(range(len(plot_data)))
        ax1.pie(
            plot_data.values(),
            labels=[k.capitalize() for k in plot_data.keys()],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax1.set_title('Daily Energy Cost Distribution', fontsize=14, fontweight='bold')

        # Bar chart
        apps = [k.capitalize() for k in plot_data.keys()]
        values = list(plot_data.values())

        ax2.bar(apps, values, color=colors)
        ax2.set_ylabel('Cost ($/day)', fontsize=11)
        ax2.set_title('Daily Cost by Appliance', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Rotate labels if needed
        if len(apps) > 3:
            ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Cost breakdown plot saved to {save_path}")

        return fig
