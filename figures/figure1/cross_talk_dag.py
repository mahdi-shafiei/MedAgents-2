"""Cross-talk DAG Visualization - Figure 1"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
try:
    from .plot_utils import get_figure_1_colors, apply_medagents_style
except ImportError:
    from plot_utils import get_figure_1_colors, apply_medagents_style

def plot_cross_talk_dag(ax, crosstalk_df, colors, panel_label='D'):
    """Plot cross-talk directed acyclic graph showing communication patterns between agents"""
    
    agents = list(crosstalk_df.columns)
    n_agents = len(agents)
    
    positions = {
        'moderator': (0.5, 0.9),
        'expert1': (0.1, 0.7),
        'expert2': (0.9, 0.7),
        'expert3': (0.5, 0.45),
        'triage': (0.5, 0.1)
    }

    node_colors = {
        'moderator': colors['agents']['moderator'],
        'triage': colors['agents']['triage'],
        'expert1': colors['agents']['expert1'],
        'expert2': colors['agents']['expert2'],
        'expert3': colors['agents']['expert3']
    }
    
    node_radius = 0.1
    for agent in agents:
        x, y = positions[agent]
        circle = plt.Circle((x, y), node_radius, facecolor=node_colors[agent], 
                          alpha=0.9, zorder=10, linewidth=3, linestyle='--', edgecolor='black')
        ax.add_patch(circle)
        
        total_interaction = crosstalk_df.loc[agent].sum()
        total_possible_interaction = crosstalk_df.sum().sum() - crosstalk_df.loc[agent, agent]
        involvement_percentage = (total_interaction / total_possible_interaction) * 100 if total_possible_interaction > 0 else 0
        
        ax.text(x, y, f'{involvement_percentage:.1f}%', 
               ha='center', va='center', fontsize=10, fontweight='bold', 
               color='white', zorder=11)
    
    for source_idx, source in enumerate(agents):
        for target_idx, target in enumerate(agents):
            if source != target:
                weight = crosstalk_df.loc[source, target]
                if weight > 0.01:
                    x1, y1 = positions[source]
                    x2, y2 = positions[target]
                    
                    dx = x2 - x1
                    dy = y2 - y1
                    length = np.sqrt(dx**2 + dy**2)
                    
                    start_offset = node_radius + 0.01
                    end_offset = node_radius + 0.01
                    
                    x1_adj = x1 + (dx / length) * start_offset
                    y1_adj = y1 + (dy / length) * start_offset
                    x2_adj = x2 - (dx / length) * end_offset
                    y2_adj = y2 - (dy / length) * end_offset
                    
                    arrow_width = 2.0
                    alpha = 0.7
                    
                    if abs(dx) > 0.1:
                        connectionstyle = "arc3,rad=0.1"
                    else:
                        connectionstyle = "arc3,rad=0"
                    
                    ax.annotate('', xy=(x2_adj, y2_adj), xytext=(x1_adj, y1_adj),
                              arrowprops=dict(arrowstyle='->', lw=arrow_width, 
                                            color='#2E2E2E', alpha=alpha,
                                            shrinkA=0, shrinkB=0,
                                            connectionstyle=connectionstyle))
                    
                    if weight > 0.15:
                        mid_x = (x1 + x2) / 2
                        mid_y = (y1 + y2) / 2
                        
                        if abs(dx) > 0.1:
                            mid_y += 0.03 if dx > 0 else -0.03
                        
                        ax.text(mid_x, mid_y, f'{weight:.2f}', 
                               ha='center', va='center', fontsize=9, 
                               bbox=dict(boxstyle='round,pad=0.15', 
                                       facecolor='white', alpha=0.9,
                                       edgecolor='gray', linewidth=0.5),
                               zorder=12)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    legend_elements = []
    agent_labels = {
        'moderator': 'Moderator',
        'triage': 'Triage Agent',
        'expert1': 'Expert 1',
        'expert2': 'Expert 2',
        'expert3': 'Expert 3'
    }
    
    for agent in agents:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=node_colors[agent], 
                                        markersize=10, label=agent_labels[agent],
                                        markeredgecolor='black', markeredgewidth=2))
    
    ax.legend(handles=legend_elements, fontsize=12, loc='center', ncol=3,
             frameon=True, fancybox=True, shadow=True, 
             framealpha=1.0, facecolor='white', edgecolor='black',
             bbox_to_anchor=(0.5, -0.1))
    
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=25, fontweight='bold')
    
    return ax

def main():
    """Example usage when run as standalone script"""
    apply_medagents_style()
    colors = get_figure_1_colors()
    
    crosstalk_df = pd.read_csv('../crosstalk.csv', index_col=0)
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_cross_talk_dag(ax, crosstalk_df, colors, panel_label='D')
    plt.tight_layout()
    plt.savefig('cross_talk_dag_example.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()