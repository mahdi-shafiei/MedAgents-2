"""Comprehensive Search Scatter Plot - Figure 2d"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    from .plot_utils import get_manchester_colors, get_figure_2_colors, apply_medagents_style
except ImportError:
    from plot_utils import get_manchester_colors, get_figure_2_colors, apply_medagents_style

def plot_comprehensive_search_scatter(ax, df, colors, panel_label='D'):
    """Create comprehensive search performance scatter plot"""
    
    # Combine all search experiments
    search_df = df[df['ablation'].isin(['search_modality', 'search_features', 'search_history'])]
    
    if not search_df.empty:
        # Create scatter plot with accuracy vs time, bubble size = cost
        for _, row in search_df.iterrows():
            color = colors['modality'].get(row['exp_name'], colors['metrics']['accuracy'])
            bubble_size = (row['avg_cost'] * 100) * 20  # Scale cost for bubble size
            
            ax.scatter(row['avg_time'], row['accuracy'], 
                      s=bubble_size, c=color, alpha=0.7,
                      edgecolors='black', linewidths=1, 
                      label=row['exp_name'])
            
            # Add experiment name annotation
            ax.annotate(row['exp_name'].replace('_', ' ').title(), 
                       (row['avg_time'], row['accuracy']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, ha='left')
    else:
        # Sample comprehensive data
        experiments = ['Both', 'Vector Only', 'Web Only', 'None', 'Baseline', 'No Doc Review']
        times = [297, 376, 75, 23, 461, 376]
        accuracies = [34, 31, 30, 25, 33, 31]
        costs = [1.26, 1.74, 0.5, 0.17, 3.72, 1.74]
        
        for i, (exp, time, acc, cost) in enumerate(zip(experiments, times, accuracies, costs)):
            bubble_size = cost * 100
            color = colors['metrics']['accuracy'] if i < 3 else colors['metrics']['time']
            
            ax.scatter(time, acc, s=bubble_size, c=color, alpha=0.7,
                      edgecolors='black', linewidths=1)
            ax.annotate(exp, (time, acc), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, ha='left')

    # Styling
    ax.set_xlabel('Average Time (seconds)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=11)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim(20, 40)
    
    # Add bubble size legend
    for size, label in [(50, '0.5¢'), (200, '2.0¢'), (400, '4.0¢')]:
        ax.scatter([], [], s=size, c='gray', alpha=0.7, edgecolors='black', linewidths=1,
                  label=f'{label} cost')
    
    ax.legend(loc='upper left', fontsize=8, title='Cost per Query', title_fontsize=9)
    
    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=15, fontweight='bold')
    
    return ax

def main():
    """Example usage when run as standalone script"""
    apply_medagents_style()
    colors = get_figure_2_colors()
    
    # Load real data
    try:
        df = pd.read_csv('../search_ablation.csv')
        print(f"Loaded {len(df)} search ablation records from real data")
    except FileNotFoundError:
        print("Warning: search_ablation.csv not found, creating sample data")
        df = pd.DataFrame({
            'ablation': ['search_modality', 'search_modality', 'search_modality', 'search_modality',
                        'search_features', 'search_features'],
            'exp_name': ['both', 'vector_only', 'web_only', 'none', 'baseline', 'no_document_review'],
            'accuracy': [34, 31, 30, 25, 33, 31],
            'avg_time': [297, 376, 75, 23, 461, 376],
            'avg_cost': [1.26, 1.74, 0.5, 0.17, 3.72, 1.74]
        })
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_comprehensive_search_scatter(ax, df, colors, panel_label='D')
    plt.tight_layout()
    plt.savefig('comprehensive_search_scatter_example.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()