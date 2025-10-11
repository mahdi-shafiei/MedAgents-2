"""Figure 0: Main Comparison - Modular with Subfigures"""
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add figure0 directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'figure0'))

from figure0.plot_utils import apply_medagents_style, get_figure_0_colors
# from figure0.main_figure_display import plot_main_figure_pdf
from figure0.overall_performance_bar import plot_overall_performance_bar
from figure0.pareto_frontier import plot_pareto_frontier
from figure0.dataset_bubble_chart import plot_dataset_bubble_chart
from figure0.dataset_beeswarm_plot import plot_dataset_beeswarm
from figure0.component_breakdown_waterfall import plot_component_breakdown_waterfall

def create_figure_0():
    """Create Figure 0: Main Comparison"""
     
    apply_medagents_style()
    
    colors = get_figure_0_colors()

    df = pd.read_csv('main_comparison.csv')
    print(f"Loaded {len(df)} records from real data")
    

    agent_df = pd.read_csv('agent_configuration.csv')
    role_play_df = pd.read_csv('role_play.csv')
    orchestration_df = pd.read_csv('orchestration_style.csv')
    print(f"Loaded agent configuration data for component breakdown")
    
    fig = plt.figure(figsize=(18, 24))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.8, 1.3, 1], width_ratios=[1, 1],
                          hspace=0.25, wspace=0.4)

    
    ax2 = fig.add_subplot(gs[0, :])
    plot_overall_performance_bar(ax2, df, colors, panel_label='A')
    
    ax3 = fig.add_subplot(gs[1, 0])
    plot_pareto_frontier(ax3, df, colors, panel_label='B')
    
    ax4 = fig.add_subplot(gs[2, 0])
    plot_component_breakdown_waterfall(ax4, agent_df, role_play_df, orchestration_df, colors, panel_label='C')
    
    ax5 = fig.add_subplot(gs[1:, 1])
    plot_dataset_beeswarm(ax5, df, colors, panel_label='D')
    
    plt.tight_layout()
    plt.savefig('fig-2.main_comparison.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

if __name__ == "__main__":
    create_figure_0()