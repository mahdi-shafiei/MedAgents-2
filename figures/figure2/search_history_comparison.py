"""Search History Comparison Plot - Figure 2c"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    from .plot_utils import get_manchester_colors, get_figure_2_colors, apply_medagents_style
except ImportError:
    from plot_utils import get_manchester_colors, get_figure_2_colors, apply_medagents_style

def plot_search_history_comparison(ax, df, colors, panel_label='C'):
    """Compare search history strategies"""
    
    # Filter data for search history ablation
    history_df = df[df['ablation'] == 'search_history']
    
    if not history_df.empty:
        history_mapping = {
            'individual': 'Individual History',
            'shared': 'Shared History'
        }
        
        history_df['history_name'] = history_df['exp_name'].map(history_mapping)
        
        colors_list = [colors['history'].get(exp_name, colors['metrics']['accuracy']) 
                      for exp_name in history_df['exp_name']]
        
        bars = ax.bar(range(len(history_df)), history_df['accuracy'], 
                     color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add annotations
        for i, (bar, row) in enumerate(zip(bars, history_df.itertuples())):
            height = bar.get_height()
            ax.annotate(f'{row.accuracy:.0f}%', 
                       (bar.get_x() + bar.get_width()/2., height + 0.5),
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # Time annotation
            ax.annotate(f'{row.avg_time:.0f}s', 
                       (bar.get_x() + bar.get_width()/2., height/2),
                       ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        ax.set_xticks(range(len(history_df)))
        ax.set_xticklabels(history_df['history_name'], fontsize=12)
        
    else:
        # Sample data
        history_types = ['Individual History', 'Shared History']
        accuracies = [34, 35]
        times = [297, 1707]
        colors_list = [colors['history']['individual'], colors['history']['shared']]
        
        bars = ax.bar(range(len(history_types)), accuracies, color=colors_list, alpha=0.8,
                     edgecolor='black', linewidth=2)
        
        # Add annotations
        for i, (bar, acc, time) in enumerate(zip(bars, accuracies, times)):
            height = bar.get_height()
            ax.annotate(f'{acc:.0f}%', 
                       (bar.get_x() + bar.get_width()/2., height + 0.5),
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax.annotate(f'{time:.0f}s', 
                       (bar.get_x() + bar.get_width()/2., height/2),
                       ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        ax.set_xticks(range(len(history_types)))
        ax.set_xticklabels(history_types, fontsize=12)

    # Styling
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=11)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax.set_ylim(25, 40)
    
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
            'ablation': ['search_history', 'search_history'],
            'exp_name': ['individual', 'shared'],
            'accuracy': [34, 35],
            'avg_time': [297, 1707],
            'avg_cost': [1.26, 1.67]
        })
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_search_history_comparison(ax, df, colors, panel_label='C')
    plt.tight_layout()
    plt.savefig('search_history_comparison_example.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()