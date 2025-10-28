import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec

# Read the batch results Excel file
excel_path = r"S:\Shared drives\invest-health\City16\CITY\tree_cover_batch_results.xlsx"
df = pd.read_excel(excel_path)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. Cost vs Tree Cover Curve (Main relationship)
ax1 = fig.add_subplot(gs[0, :2])
for city in df['city'].unique():
    city_data = df[df['city'] == city].sort_values('target_treecover_pct')
    ax1.plot(city_data['target_treecover_pct'],
             city_data['preventable_cost_usd'] / 1000,  # Convert to thousands
             marker='o', label=f'City {city}', linewidth=2, markersize=4, alpha=0.7)

ax1.set_xlabel('Tree Cover Target (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Preventable Cost (1000 USD)', fontsize=12, fontweight='bold')
ax1.set_title('Cost Savings by Tree Cover Target - All Cities', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. Target NDVI vs Tree Cover
ax2 = fig.add_subplot(gs[0, 2])
for city in df['city'].unique():
    city_data = df[df['city'] == city].sort_values('target_treecover_pct')
    ax2.plot(city_data['target_treecover_pct'],
             city_data['target_ndvi'],
             marker='o', label=f'City {city}', linewidth=2, markersize=4, alpha=0.7)

ax2.set_xlabel('Tree Cover Target (%)', fontsize=10)
ax2.set_ylabel('Target NDVI', fontsize=10)
ax2.set_title('NDVI-Tree Cover Relationship', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=7)

# 3. Preventable Cases by City (Bar chart)
ax3 = fig.add_subplot(gs[1, 0])
# Get max preventable cases for each city
max_cases_by_city = df.groupby('city')['preventable_cases'].max().sort_values(ascending=False)
colors = sns.color_palette("viridis", len(max_cases_by_city))
ax3.barh(range(len(max_cases_by_city)), max_cases_by_city.values, color=colors)
ax3.set_yticks(range(len(max_cases_by_city)))
ax3.set_yticklabels([f'City {c}' for c in max_cases_by_city.index], fontsize=9)
ax3.set_xlabel('Max Preventable Cases', fontsize=10)
ax3.set_title('Maximum Preventable Cases by City', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# 4. Cost per Case Distribution
ax4 = fig.add_subplot(gs[1, 1])
for city in df['city'].unique():
    city_data = df[df['city'] == city]
    ax4.hist(city_data['cost_per_case_usd'] / 1000,
             bins=15, alpha=0.5, label=f'City {city}', edgecolor='black')

ax4.set_xlabel('Cost per Case (1000 USD)', fontsize=10)
ax4.set_ylabel('Frequency', fontsize=10)
ax4.set_title('Distribution of Cost per Case', fontsize=11, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# 5. Efficiency Score: Cost Savings per % Tree Cover Increase
ax5 = fig.add_subplot(gs[1, 2])
efficiency_data = []
for city in df['city'].unique():
    city_data = df[df['city'] == city].sort_values('tree_cover_increase_pct')
    if len(city_data) > 1:
        # Calculate efficiency as cost increase per tree cover increase
        max_cost = city_data['preventable_cost_usd'].max()
        max_increase = city_data['tree_cover_increase_pct'].max()
        min_increase = city_data['tree_cover_increase_pct'].min()
        if max_increase > min_increase:
            efficiency = max_cost / (max_increase - min_increase)
            efficiency_data.append({'city': city, 'efficiency': efficiency})

if efficiency_data:
    eff_df = pd.DataFrame(efficiency_data).sort_values('efficiency', ascending=False)
    colors = sns.color_palette("coolwarm", len(eff_df))
    ax5.barh(range(len(eff_df)), eff_df['efficiency'].values / 1000, color=colors)
    ax5.set_yticks(range(len(eff_df)))
    ax5.set_yticklabels([f"City {c}" for c in eff_df['city']], fontsize=9)
    ax5.set_xlabel('Cost per % Tree Cover Increase (1000 USD/%)', fontsize=10)
    ax5.set_title('Cost Efficiency by City', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')

# 6. Summary Statistics Table
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')

summary_data = []
for city in df['city'].unique():
    city_data = df[df['city'] == city]
    summary_data.append({
        'City': city,
        'Max Cost (k$)': f"{city_data['preventable_cost_usd'].max()/1000:.1f}",
        'Max Cases': f"{city_data['preventable_cases'].max():.0f}",
        'Current Tree Cover': f"{city_data['current_treecover_pct'].iloc[0]:.1f}%",
        'Target Range': f"{city_data['target_treecover_pct'].min():.1f}-{city_data['target_treecover_pct'].max():.1f}%",
        'NDVI Range': f"{city_data['target_ndvi'].min():.3f}-{city_data['target_ndvi'].max():.3f}",
        'Samples': len(city_data)
    })

summary_df = pd.DataFrame(summary_data)
table = ax6.table(cellText=summary_df.values, colLabels=summary_df.columns,
                  cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style the table
for i in range(len(summary_df.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(summary_df) + 1):
    for j in range(len(summary_df.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

ax6.set_title('Summary Statistics by City', fontsize=14, fontweight='bold', pad=20)

plt.suptitle('Batch Analysis Results - Mental Health Cost Savings by Tree Cover',
             fontsize=16, fontweight='bold', y=0.98)

# Save figure
output_path = r'S:\Shared drives\invest-health\City16\CITY\\batch_analysis_visualization.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Visualization saved to: {output_path}")

plt.show()

# Print detailed summary
print("\n" + "="*70)
print("DETAILED SUMMARY STATISTICS")
print("="*70)

for city in df['city'].unique():
    city_data = df[df['city'] == city]
    print(f"   CITY: {city}")
    print(f"   Total samples: {len(city_data)}")
    print(f"   Current tree cover: {city_data['current_treecover_pct'].iloc[0]:.1f}%")
    print(f"   Tree cover increase range: {city_data['tree_cover_increase_pct'].min():.1f}% - {city_data['tree_cover_increase_pct'].max():.1f}%")
    print(f"   Target tree cover range: {city_data['target_treecover_pct'].min():.1f}% - {city_data['target_treecover_pct'].max():.1f}%")
    print(f"   Target NDVI range: {city_data['target_ndvi'].min():.3f} - {city_data['target_ndvi'].max():.3f}")
    print(f"   Max preventable cases: {city_data['preventable_cases'].max():,.0f}")
    print(f"   Max cost savings: ${city_data['preventable_cost_usd'].max():,.0f}")
    print(f"   Mean cost savings: ${city_data['preventable_cost_usd'].mean():,.0f}")
    print(f"   Cost per case: ${city_data['cost_per_case_usd'].mean():,.0f}")
    print(f"   Max tract cases: {city_data['max_tract_cases'].max():.0f}" if 'max_tract_cases' in city_data.columns else "")

print("\n" + "="*70)