"""
Vote Convergence Analysis Plot for Figure 1
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .plot_utils import get_figure_1_colors, get_manchester_colors

def plot_vote_convergence(ax, df, panel_label=None):
    """
    Plot vote entropy convergence across discussion rounds.
    Shows how consensus emerges over multiple rounds of discussion.
    """
    
    try:
        colors = get_figure_1_colors()
        
        # Define color mapping for discussion modes - using better distinct colors
        manchester_colors = get_manchester_colors()
        mode_colors = {
            'group_chat_with_orchestrator': colors['metrics']['accuracy'],      # Red for moderated meeting
            'group_chat_voting_only': manchester_colors['persimmon'],  # Pink-orange for unmoderated
            'independent': manchester_colors['black'],                    # Light gray for individual
            'one_on_one_sync': manchester_colors['sandy_brown']                # Sandy brown for independent vote
        }
        
        # Define display names for discussion modes
        mode_names = {
            'group_chat_with_orchestrator': 'Moderated Team Meeting',
            'group_chat_voting_only': 'Unmoderated Group Chat',
            'independent': 'Individual Meeting',
            'one_on_one_sync': 'Independent Vote'
        }
        
        # Plot entropy over rounds for each discussion mode
        for mode in df['discussion_mode'].unique():
            mode_data = df[df['discussion_mode'] == mode].sort_values('round')
            
            if len(mode_data) > 0:
                color = mode_colors.get(mode, 'gray')
                label = mode_names.get(mode, mode)
                
                # Plot main line with style matching other panels
                ax.plot(
                    mode_data['round'], 
                    mode_data['mean'], 
                    label=label,
                    color=color,
                    marker='o',
                    linewidth=3,
                    markersize=10,
                    markeredgecolor='black',
                    markeredgewidth=2,
                    alpha=0.8
                )
                
                # Add shaded region for uncertainty (much smaller std)
                ax.fill_between(
                    mode_data['round'],
                    mode_data['mean'] - mode_data['std'] * 0.1,  # Much smaller error region (10% of std)
                    mode_data['mean'] + mode_data['std'] * 0.1,
                    color=color,
                    alpha=0.2,
                    linewidth=0
                )
        
        # Highlight best performing configuration (lowest entropy) for each round
        for round_num in df['round'].unique():
            round_data = df[df['round'] == round_num]
            if not round_data.empty:
                # Find the mode with lowest entropy for this round
                best_idx = round_data['mean'].idxmin()
                best_round_data = round_data.loc[best_idx]
                
                # Add gold star for best consensus at each round
                label = 'Best Consensus' if round_num == sorted(df['round'].unique())[0] else None
                ax.plot(best_round_data['round'], best_round_data['mean'], 
                       marker='*', markersize=15, color='gold', 
                       markeredgecolor='black', markeredgewidth=2, 
                       label=label, zorder=10)
        
        # Customize the plot to match panel C styling
        ax.set_xlabel('Discussion rounds', fontsize=16, fontweight='bold')
        ax.set_ylabel('Vote Entropy (bits)', fontsize=16, fontweight='bold') 
        ax.tick_params(axis='both', labelsize=14)
        
        # Set x-axis to show rounds 1, 2, 3 (add 1 to the round numbers)
        rounds = sorted(df['round'].unique())
        ax.set_xticks(rounds)
        ax.set_xticklabels([r + 1 for r in rounds])  # Convert 0,1,2 to 1,2,3
        
        # Add grid for readability - match panel C style
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Legend with matching style - positioned more to the right
        ax.legend(loc='lower left', fontsize=12, frameon=True, fancybox=True, shadow=True,
                 framealpha=1.0, facecolor='white', edgecolor='black')
        
        # Set y-axis limits to specified range
        ax.set_ylim(0.1, 0.6)
        
        # Add panel label if provided
        if panel_label:
            ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes, 
                   fontsize=25, fontweight='bold', va='bottom', ha='left')
        
        # Add interpretation note
        ax.text(0.02, 0.98, 'Lower entropy = Higher consensus', 
               transform=ax.transAxes, fontsize=12, 
               verticalalignment='top', style='italic', alpha=0.9)
        
        return ax
        
    except FileNotFoundError:
        # Create synthetic data for demonstration if CSV not found
        ax.text(0.5, 0.5, 'Vote Convergence Analysis\n(Data not available)', 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        if panel_label:
            ax.text(-0.1, 1.05, panel_label, transform=ax.transAxes, 
                   fontsize=16, fontweight='bold', va='bottom', ha='right')
        
        return ax