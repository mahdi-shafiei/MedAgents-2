"""Figure 2: Search Ablation Analysis - Modular with Subfigures"""
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add figure2 directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'figure2'))

from figure2.plot_utils import apply_medagents_style, get_figure_2_colors
from figure2.search_modality_comparison import plot_search_modality_comparison
from figure2.search_features_comparison import plot_search_features_comparison
from figure2.search_history_comparison import plot_search_history_comparison
from figure2.comprehensive_search_scatter import plot_comprehensive_search_scatter

def create_figure_2():
    """Create Figure 2: Search Ablation Analysis"""
    
    # Load original MedAgents style
    apply_medagents_style()
    
    # Get color scheme
    colors = get_figure_2_colors()
    
    # Load data
    try:
        df = pd.read_csv('search_ablation.csv')
        print(f"Loaded {len(df)} search ablation records from real data")
    except FileNotFoundError:
        print("Warning: search_ablation.csv not found, creating sample data")
        df = pd.DataFrame({
            'ablation': ['search_modality', 'search_modality', 'search_modality', 'search_modality',
                        'search_features', 'search_features', 'search_features', 'search_features',
                        'search_history', 'search_history'],
            'exp_name': ['both', 'vector_only', 'web_only', 'none',
                        'baseline', 'no_document_review', 'no_query_rewrite', 'no_rewrite_no_review',
                        'individual', 'shared'],
            'accuracy': [34, 31, 30, 25, 33, 31, 27, 29, 34, 35],
            'avg_time': [297, 376, 75, 23, 461, 376, 548, 291, 297, 1707],
            'avg_cost': [1.26, 1.74, 0.5, 0.17, 3.72, 1.74, 3.73, 1.0, 1.26, 1.67]
        })
    
    # Create figure and grid
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1.5], 
                          hspace=0.4, wspace=0.5)
    
    # Subplot a: Search modality comparison
    ax1 = fig.add_subplot(gs[0, 0])
    plot_search_modality_comparison(ax1, df, colors, panel_label='A')
    
    # Subplot b: Search features comparison
    ax2 = fig.add_subplot(gs[0, 1])
    plot_search_features_comparison(ax2, df, colors, panel_label='B')
    
    # Subplot c: Search history comparison
    ax3 = fig.add_subplot(gs[1, 0]) 
    plot_search_history_comparison(ax3, df, colors, panel_label='C')
    
    # Subplot d: Comprehensive search scatter
    ax4 = fig.add_subplot(gs[1, 1])
    plot_comprehensive_search_scatter(ax4, df, colors, panel_label='D')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('fig-4.search_ablation.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

if __name__ == "__main__":
    create_figure_2()