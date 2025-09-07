"""Search Modality Comparison - Figure 2a"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    from .plot_utils import get_figure_2_colors, apply_medagents_style
except ImportError:
    from plot_utils import get_figure_2_colors, apply_medagents_style

def plot_search_modality_comparison(ax, df, colors, panel_label='A'):
    """Compare different search modalities (web vs vector vs both)"""
    
    # Configuration
    modality_names = ['Search with\\nWeb & Vector', 'Search with\\nVector Only', 
                     'Search with\\nWeb Only', 'No Search', 'Random']
    
    # Filter data for search modality ablation
    modality_df = df[df['ablation'] == 'search_modality']
    
    if not modality_df.empty:
        accuracies = modality_df['accuracy'].values
        times = modality_df['avg_time'].values
        costs = modality_df['avg_cost'].values

        # Create dual-axis bar chart
        bars1 = ax.bar(np.arange(len(modality_names)) - 0.2, accuracies, 0.4, 
                       color=colors['metrics']['accuracy'], alpha=0.8, 
                       label='Accuracy (%)', edgecolor='black', linewidth=1.5)

        ax_twin = ax.twinx()
        bars2 = ax_twin.bar(np.arange(len(modality_names)) + 0.2, times, 0.4,
                           color=colors['metrics']['time'], alpha=0.8,
                           label='Time (s)', edgecolor='black', linewidth=1.5)

        # Add annotations
        best_accuracy = accuracies[0]
        for i, (bar, acc, cost) in enumerate(zip(bars1, accuracies, costs)):
            height = bar.get_height()
            # Accuracy percentage on top
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{acc:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
            # Cost in center of bar
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{cost:.1f}¢', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
            
            # Performance reduction annotation for non-best methods
            if i > 0:
                reduction = ((best_accuracy - acc) / best_accuracy) * 100
                ax.text(bar.get_x() + bar.get_width()/2., height + 3,
                       f'↓{reduction:.0f}%', ha='center', va='bottom', fontsize=10, 
                       fontweight='bold', color=colors['annotations']['text_box_edge'], 
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor=colors['annotations']['text_box_face'], alpha=0.7))

        # Time annotations
        for i, (bar, time) in enumerate(zip(bars2, times)):
            height = bar.get_height()
            ax_twin.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{time:.0f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
                 bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)
    else:
        # Use sample data if no real data available
        accuracies = [34, 31, 30, 25, 23]
        times = [297, 376, 75, 23]
        costs = [1.26, 1.74, 0.5, 0.17]
        
        bars1 = ax.bar(np.arange(len(modality_names)) - 0.2, accuracies, 0.4, 
                       color=colors['metrics']['accuracy'], alpha=0.8, 
                       label='Accuracy (%)', edgecolor='black', linewidth=1.5)

        ax_twin = ax.twinx()
        bars2 = ax_twin.bar(np.arange(len(modality_names)) + 0.2, times, 0.4,
                           color=colors['metrics']['time'], alpha=0.8,
                           label='Time (s)', edgecolor='black', linewidth=1.5)

    # Styling
    ax.set_xlabel('Search Modality', fontweight='bold', fontsize=11, color='black')
    ax.set_ylabel('Accuracy (%)', fontweight='bold', color='black', fontsize=11)
    ax_twin.set_ylabel('Time (s)', fontweight='bold', color='black', fontsize=11)
    ax.set_xticks(np.arange(len(modality_names)))
    ax.set_xticklabels(modality_names, fontsize=10)
    ax.tick_params(axis='both', labelsize=10, colors='black')
    ax_twin.tick_params(axis='both', labelsize=10, colors='black')
    ax.set_ylim(0, 40)
    ax_twin.set_ylim(0, 400)
    
    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=15, fontweight='bold')
    
    return ax, ax_twin

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
            'ablation': ['search_modality', 'search_modality', 'search_modality', 'search_modality'],
            'exp_name': ['both', 'vector_only', 'web_only', 'none', 'random'],
            'accuracy': [34, 31, 30, 25, 23],
            'avg_time': [297, 376, 75, 23],
            'avg_cost': [1.26, 1.74, 0.5, 0.17, 0.14]
        })
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_search_modality_comparison(ax, df, colors, panel_label='A')
    plt.tight_layout()
    plt.savefig('search_modality_comparison_example.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()