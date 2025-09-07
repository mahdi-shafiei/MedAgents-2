"""Pareto Frontier Plot - Figure 0b"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import get_manchester_colors, get_figure_0_colors, apply_medagents_style

def plot_pareto_frontier(ax, df, colors, panel_label):
    """Create Pareto frontier cost vs accuracy plot"""
    
    method_colors = colors.get('methods', {})
    
    # Calculate average metrics
    avg_metrics = df.groupby('method').agg({
        'accuracy': 'mean',
        'avg_cost': lambda x: x.mean() * 100,  # Convert to cents
        'avg_time': 'mean'
    }).reset_index()

    # Plot scatter points
    for _, row in avg_metrics.iterrows():
        if row['method'] == 'MedAgents-2':
            ax.scatter(row['avg_cost'], row['accuracy'], 
                      c=method_colors.get(row['method'], 'gray'), 
                      s=500, alpha=0.9, 
                      edgecolors='black', linewidth=3,
                      label=row['method'], marker='*', zorder=10)
        else:
            ax.scatter(row['avg_cost'], row['accuracy'], 
                      c=method_colors.get(row['method'], 'gray'), 
                      s=250, alpha=0.8, 
                      edgecolors='black', linewidth=2,
                      label=row['method'], zorder=5)

    # Calculate Pareto frontier
    pareto_indices = []
    sorted_indices = np.argsort(avg_metrics['avg_cost'])
    max_accuracy_so_far = -1
    for i in sorted_indices:
        if avg_metrics.iloc[i]['accuracy'] > max_accuracy_so_far:
            pareto_indices.append(i)
            max_accuracy_so_far = avg_metrics.iloc[i]['accuracy']

    if len(pareto_indices) > 1:
        pareto_data = avg_metrics.iloc[pareto_indices].sort_values('avg_cost')
        ax.plot(pareto_data['avg_cost'], pareto_data['accuracy'], 'k--', alpha=0.8, linewidth=3, 
               label='Pareto Frontier', zorder=8)

    # Add annotations
    for _, row in avg_metrics.iterrows():
        offset_x = 12 if row['method'] != 'MedAgents-2' else 15
        offset_y = 10 if row['method'] != 'MedAgents-2' else 12
        ax.annotate(row['method'], (row['avg_cost'], row['accuracy']), 
                   xytext=(offset_x, offset_y), textcoords='offset points', 
                   fontsize=13, ha='left', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                            edgecolor='gray', linewidth=1.5))

    # Styling
    ax.set_xlabel('Average Cost (cents per query)', fontweight='bold', fontsize=18)
    ax.set_ylabel('Average Accuracy (%)', fontweight='bold', fontsize=18)
    ax.set_ylim(14, 34)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, alpha=0.4, linestyle=':', linewidth=1)
    
    # Legend
    pareto_legend = [plt.Line2D([0], [0], color='black', linestyle='--', linewidth=3, 
                               label='Pareto Frontier', alpha=0.8)]
    ax.legend(handles=pareto_legend, loc='lower right', fontsize=13, 
             frameon=True, fancybox=True, shadow=True, framealpha=0.95,
             facecolor='white', edgecolor='black')
    
    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=25, fontweight='bold')
    
    return ax

def main():
    """Example usage when run as standalone script"""
    apply_medagents_style()
    colors = get_figure_0_colors()
    
    # Load real data
    try:
        df = pd.read_csv('../main_comparison.csv')
        print(f"Loaded {len(df)} records from real data")
    except FileNotFoundError:
        print("Warning: main_comparison.csv not found, creating sample data")
        df = pd.DataFrame({
            'method': ['MedAgents-2', 'CoT', 'Zero-shot', 'Few-shot'],
            'accuracy': [35.2, 28.1, 18.5, 24.7],
            'avg_cost': [0.125, 0.085, 0.023, 0.055],
            'avg_time': [65.3, 42.1, 15.2, 28.3]
        })
    
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_pareto_frontier(ax, df, colors, panel_label='B')
    plt.tight_layout()
    plt.savefig('pareto_frontier_example.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()