"""Supplemental Figure S0: GraphPad Prism Style Dataset Comparison - Modular with Subfigures"""
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add figures0 directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'figures0'))

from plot_style import set_plot_style, get_color_scheme
from figures0.graphpad_prism_bars import plot_graphpad_prism_bars
from figures0.pareto_frontier_models import plot_pareto_frontier_models

def create_figures():
    """Create Supplemental Figure S0: Dataset Comparison"""
    
    # Load original style
    set_plot_style()
    colors = get_color_scheme('figure_0', theme='manchester_united_official')
    
    # Load real data
    try:
        raw = pd.read_csv('main_comparison.csv')
        print(f"Loaded {len(raw)} records from real data")
    except FileNotFoundError:
        print("Warning: main_comparison.csv not found, creating sample data")
        # Create comprehensive sample data
        methods = ['MedAgents-2', 'CoT', 'Zero-shot', 'Few-shot'] * 9
        datasets = ['medbullets', 'medqa', 'pubmedqa'] * 12
        models = ['o3-mini'] * 12 + ['gpt-4o'] * 12 + ['gpt-4o-mini'] * 12
        raw = pd.DataFrame({
            'method': methods,
            'dataset': datasets,
            'model': models,
            'accuracy': [42.5, 35.2, 18.5, 28.3] * 9,
            'avg_time': [65.3, 42.1, 15.2, 31.7] * 9,
            'avg_cost': [0.125, 0.085, 0.023, 0.067] * 9
        })
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
    
    # Subfigure A: GraphPad Prism style bar plots (3x3)
    ax_bars = fig.add_subplot(gs[0, 0])
    fig_bars, axes_bars = plot_graphpad_prism_bars(ax_bars, raw, colors, panel_label='A')
    ax_bars.set_visible(False)  # Hide the container axis
    
    # Subfigure B: Pareto frontier by models (3x1)  
    ax_pareto = fig.add_subplot(gs[1, 0])
    fig_pareto, axes_pareto = plot_pareto_frontier_models(raw, colors, panel_label='B')
    ax_pareto.set_visible(False)  # Hide the container axis
    
    # Save figure
    plt.tight_layout()
    plt.savefig('fig-s0.dataset_comparison.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig


if __name__ == '__main__':
    create_figures()
