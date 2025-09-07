"""Figure 0: Main Comparison - Modular with Subfigures"""
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add figure0 directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'figure0'))

from figure0.plot_utils import apply_medagents_style, get_figure_0_colors
from figure0.main_figure_display import plot_main_figure_pdf
from figure0.overall_performance_bar import plot_overall_performance_bar
from figure0.pareto_frontier import plot_pareto_frontier
from figure0.dataset_bubble_chart import plot_dataset_bubble_chart
from figure0.dataset_beeswarm_plot import plot_dataset_beeswarm
from figure0.component_breakdown_waterfall import plot_component_breakdown_waterfall

def create_figure_0():
    """Create Figure 0: Main Comparison"""
    
    # Load original MedAgents style
    apply_medagents_style()
    
    # Get color scheme
    colors = get_figure_0_colors()
    
    # Load real data
    try:
        df = pd.read_csv('main_comparison.csv')
        print(f"Loaded {len(df)} records from real data")
    except FileNotFoundError:
        print("Warning: main_comparison.csv not found, creating sample data")
        # Create comprehensive sample data
        methods = ['MedAgents-2', 'CoT', 'Zero-shot', 'Few-shot'] * 9
        datasets = ['medbullets', 'medqa', 'pubmedqa'] * 12
        models = ['o3-mini'] * 12 + ['gpt-4o'] * 12 + ['gpt-4o-mini'] * 12
        df = pd.DataFrame({
            'method': methods,
            'dataset': datasets,
            'model': models,
            'accuracy': [42.5, 35.2, 18.5, 28.3] * 9,
            'avg_time': [65.3, 42.1, 15.2, 31.7] * 9,
            'avg_cost': [0.125, 0.085, 0.023, 0.067] * 9
        })
    
    # Load additional data for component breakdown
    try:
        agent_df = pd.read_csv('agent_configuration.csv')
        role_play_df = pd.read_csv('role_play.csv')
        orchestration_df = pd.read_csv('orchestration_style.csv')
        print(f"Loaded agent configuration data for component breakdown")
    except FileNotFoundError:
        print("Warning: component breakdown data not found, creating sample data")
        agent_df = pd.DataFrame({
            'exp_name': ['1_agent_1_round_no_search', '3_agents_1_round_no_search', '3_agents_1_round_with_search', '3_agents_3_rounds_with_search'],
            'accuracy': [20, 25, 30, 32]
        })
        role_play_df = pd.DataFrame({
            'exp_name': ['enable_role_play', 'disable_role_play'],
            'accuracy': [35, 30]
        })
        orchestration_df = pd.DataFrame({
            'exp_name': ['group_chat_with_orchestrator', 'independent'],
            'accuracy': [35, 28]
        })
    
    # Create figure and grid
    fig = plt.figure(figsize=(24, 30))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.8, 1, 1, 1], width_ratios=[1, 1],
                          hspace=0.25, wspace=0.4)
    
    # Subplot a: Main figure PDF display
    ax1 = fig.add_subplot(gs[0, :])
    plot_main_figure_pdf(ax1, panel_label='A')
    
    # Subplot b: Overall performance bar chart
    ax2 = fig.add_subplot(gs[1, :])
    plot_overall_performance_bar(ax2, df, colors, panel_label='B')
    
    # Bottom row with 1:1:1 width ratio
    # Subplot c: Pareto frontier (bottom left, 1/3 width)
    ax3 = fig.add_subplot(gs[2, 0])
    plot_pareto_frontier(ax3, df, colors, panel_label='C')
    
    # Subplot d: Component breakdown waterfall (bottom middle, 1/3 width)
    ax4 = fig.add_subplot(gs[3, 0])
    plot_component_breakdown_waterfall(ax4, agent_df, role_play_df, orchestration_df, colors, panel_label='D')
    
    # Subplot e: Dataset beeswarm plot (bottom right, 1/3 width)
    ax5 = fig.add_subplot(gs[2:, 1])
    plot_dataset_beeswarm(ax5, df, colors, panel_label='E')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('fig-2.main_comparison.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

if __name__ == "__main__":
    create_figure_0()