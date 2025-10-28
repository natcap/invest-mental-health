import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def create_city_visualizations(excel_path, output_dir):
    """Create various visualizations from the batch processing results"""

    # Read the data
    df = pd.read_excel(excel_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get unique cities
    cities = df['city'].unique()

    print(f"Creating visualizations for {len(cities)} cities...")

    # 1. Individual city curves
    create_individual_city_plots(df, cities, output_dir)

    # 2. All cities comparison
    create_comparison_plots(df, cities, output_dir)

    # 3. Summary statistics plots
    create_summary_plots(df, output_dir)

    # 4. Cost-benefit analysis plots
    create_cost_benefit_plots(df, output_dir)

    print(f"All visualizations saved to: {output_dir}")


def create_individual_city_plots(df, cities, output_dir):
    """Create individual plots for each city"""

    city_dir = os.path.join(output_dir, "individual_cities")
    os.makedirs(city_dir, exist_ok=True)

    for city in cities:
        city_data = df[df['city'] == city].sort_values('tree_cover_increase_pct')

        # Create a 2x2 subplot for each city
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{city} - Tree Cover Analysis', fontsize=16, fontweight='bold')

        # 1. Preventable Cases vs Tree Cover Increase
        ax1.plot(city_data['tree_cover_increase_pct'], city_data['preventable_cases'],
                 'o-', linewidth=2, markersize=6, color='darkgreen')
        ax1.set_xlabel('Tree Cover Increase (%)')
        ax1.set_ylabel('Preventable Cases')
        ax1.set_title('Preventable Cases vs Tree Cover Increase')
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='plain', axis='y')

        # Add value labels on key points
        for i in range(0, len(city_data), 5):  # Every 5th point
            row = city_data.iloc[i]
            ax1.annotate(f'{int(row["preventable_cases"] / 1000)}K',
                         (row['tree_cover_increase_pct'], row['preventable_cases']),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

        # 2. Cost Savings vs Tree Cover Increase
        ax2.plot(city_data['tree_cover_increase_pct'], city_data['preventable_cost_usd'] / 1e9,
                 'o-', linewidth=2, markersize=6, color='darkblue')
        ax2.set_xlabel('Tree Cover Increase (%)')
        ax2.set_ylabel('Cost Savings (Billion $)')
        ax2.set_title('Cost Savings vs Tree Cover Increase')
        ax2.grid(True, alpha=0.3)

        # 3. Target NDVI vs Tree Cover
        ax3.plot(city_data['target_treecover_pct'], city_data['target_ndvi'],
                 'o-', linewidth=2, markersize=6, color='darkred')
        ax3.set_xlabel('Target Tree Cover (%)')
        ax3.set_ylabel('Target NDVI')
        ax3.set_title('NDVI vs Tree Cover Relationship')
        ax3.grid(True, alpha=0.3)

        # 4. Max Tract Cases vs Tree Cover Increase
        ax4.plot(city_data['tree_cover_increase_pct'], city_data['max_tract_cases'],
                 'o-', linewidth=2, markersize=6, color='darkorange')
        ax4.set_xlabel('Tree Cover Increase (%)')
        ax4.set_ylabel('Max Tract Cases')
        ax4.set_title('Maximum Tract Cases vs Tree Cover Increase')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(city_dir, f'{city}_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()


def create_comparison_plots(df, cities, output_dir):
    """Create comparison plots showing all cities together"""

    # 1. All cities - Preventable Cases
    plt.figure(figsize=(14, 10))
    for city in cities:
        city_data = df[df['city'] == city].sort_values('tree_cover_increase_pct')
        plt.plot(city_data['tree_cover_increase_pct'], city_data['preventable_cases'],
                 'o-', linewidth=2, markersize=4, label=city, alpha=0.8)

    plt.xlabel('Tree Cover Increase (%)', fontsize=12)
    plt.ylabel('Preventable Cases', fontsize=12)
    plt.title('Preventable Cases vs Tree Cover Increase - All Cities', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style='plain', axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_cities_preventable_cases.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. All cities - Cost Savings
    plt.figure(figsize=(14, 10))
    for city in cities:
        city_data = df[df['city'] == city].sort_values('tree_cover_increase_pct')
        plt.plot(city_data['tree_cover_increase_pct'], city_data['preventable_cost_usd'] / 1e9,
                 'o-', linewidth=2, markersize=4, label=city, alpha=0.8)

    plt.xlabel('Tree Cover Increase (%)', fontsize=12)
    plt.ylabel('Cost Savings (Billion $)', fontsize=12)
    plt.title('Cost Savings vs Tree Cover Increase - All Cities', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_cities_cost_savings.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_plots(df, output_dir):
    """Create summary statistics plots"""

    # 1. Current Tree Cover by City
    city_baseline = df.groupby('city')['current_treecover_pct'].first().sort_values()

    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(city_baseline)), city_baseline.values, color='forestgreen', alpha=0.7)
    plt.xlabel('Cities', fontsize=12)
    plt.ylabel('Current Tree Cover (%)', fontsize=12)
    plt.title('Current Tree Cover by City', fontsize=14, fontweight='bold')
    plt.xticks(range(len(city_baseline)), city_baseline.index, rotation=45, ha='right')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'current_tree_cover_by_city.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Total Impact at 20% increase
    max_increase = df.groupby('city')['tree_cover_increase_pct'].max()
    total_impact = df[df['tree_cover_increase_pct'] == 20].copy()
    total_impact = total_impact.sort_values('preventable_cases', ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Preventable cases at 20% increase
    bars1 = ax1.barh(range(len(total_impact)), total_impact['preventable_cases'] / 1000,
                     color='steelblue', alpha=0.7)
    ax1.set_xlabel('Preventable Cases (Thousands)', fontsize=12)
    ax1.set_ylabel('Cities', fontsize=12)
    ax1.set_title('Preventable Cases at 20% Tree Cover Increase', fontsize=14, fontweight='bold')
    ax1.set_yticks(range(len(total_impact)))
    ax1.set_yticklabels(total_impact['city'])
    ax1.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2.,
                 f'{int(width)}K', ha='left', va='center', fontsize=9)

    # Cost savings at 20% increase
    bars2 = ax2.barh(range(len(total_impact)), total_impact['preventable_cost_usd'] / 1e9,
                     color='darkgreen', alpha=0.7)
    ax2.set_xlabel('Cost Savings (Billion $)', fontsize=12)
    ax2.set_title('Cost Savings at 20% Tree Cover Increase', fontsize=14, fontweight='bold')
    ax2.set_yticks(range(len(total_impact)))
    ax2.set_yticklabels(total_impact['city'])
    ax2.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2.,
                 f'${width:.1f}B', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_impact_at_20pct.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_cost_benefit_plots(df, output_dir):
    """Create cost-benefit analysis plots"""

    # 1. Cost per case prevented by city and tree cover increase
    plt.figure(figsize=(14, 10))

    # Calculate cost per case
    df['cost_per_case_actual'] = df['preventable_cost_usd'] / df['preventable_cases']

    for city in df['city'].unique():
        city_data = df[df['city'] == city].sort_values('tree_cover_increase_pct')
        plt.plot(city_data['tree_cover_increase_pct'], city_data['cost_per_case_actual'],
                 'o-', linewidth=2, markersize=4, label=city, alpha=0.8)

    plt.xlabel('Tree Cover Increase (%)', fontsize=12)
    plt.ylabel('Cost per Case Prevented ($)', fontsize=12)
    plt.title('Cost per Case Prevented vs Tree Cover Increase', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_per_case_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Efficiency analysis - Cases per % tree cover increase
    df['cases_per_pct_increase'] = df['preventable_cases'] / df['tree_cover_increase_pct']

    plt.figure(figsize=(14, 8))

    efficiency_data = df.groupby('city')['cases_per_pct_increase'].mean().sort_values(ascending=True)
    bars = plt.bar(range(len(efficiency_data)), efficiency_data.values,
                   color='purple', alpha=0.7)
    plt.xlabel('Cities', fontsize=12)
    plt.ylabel('Average Cases Prevented per 1% Tree Cover Increase', fontsize=12)
    plt.title('Tree Cover Efficiency by City', fontsize=14, fontweight='bold')
    plt.xticks(range(len(efficiency_data)), efficiency_data.index, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tree_cover_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()


# Main execution
if __name__ == "__main__":
    # File paths
    excel_file = r"S:\Shared drives\invest-health\City16\CITY\tree_cover_batch_results.xlsx"
    output_directory = r"S:\Shared drives\invest-health\City16\CITY"

    # Create visualizations
    create_city_visualizations(excel_file, output_directory)

    print("\nVisualization Summary:")
    print("=" * 50)
    print("Created the following visualization types:")
    print("1. Individual city analysis plots (2x2 subplots for each city)")
    print("2. All cities comparison plots")
    print("3. Current tree cover baseline comparison")
    print("4. Total impact analysis at 20% increase")
    print("5. Cost-benefit analysis plots")
    print("6. Tree cover efficiency analysis")
    print(f"\nAll files saved to: {output_directory}")