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
        
        manchester_colors = get_manchester_colors()
        mode_colors = {
            'group_chat_with_orchestrator': colors['metrics']['accuracy'],      
            'group_chat_voting_only': manchester_colors['persimmon'],  
            'independent': manchester_colors['black'],                    
            'one_on_one_sync': manchester_colors['sandy_brown']                
        }
        
        mode_names = {
            'group_chat_with_orchestrator': 'Moderated Team Meeting',
            'group_chat_voting_only': 'Unmoderated Group Chat',
            'independent': 'Independent Vote',
            'one_on_one_sync': 'Individual Meeting'
        }
        
        for mode in df['discussion_mode'].unique():
            mode_data = df[df['discussion_mode'] == mode].sort_values('round')
            
            if len(mode_data) > 0:
                color = mode_colors.get(mode, 'gray')
                label = mode_names.get(mode, mode)
                
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
                
                ax.fill_between(
                    mode_data['round'],
                    mode_data['mean'] - mode_data['std'] * 0.05,  
                    mode_data['mean'] + mode_data['std'] * 0.05,
                    color=color,
                    alpha=0.2,
                    linewidth=0
                )
        
        for round_num in df['round'].unique():
            round_data = df[df['round'] == round_num]
            if not round_data.empty:
                best_idx = round_data['mean'].idxmin()
                worst_idx = round_data['mean'].idxmax()
                
                best_round_data = round_data.loc[best_idx]
                worst_round_data = round_data.loc[worst_idx]
                
                ax.annotate(f'{best_round_data["mean"]:.2f}', 
                           (best_round_data['round'], best_round_data['mean']),
                           textcoords="offset points", xytext=(0,15), ha='center',
                           fontsize=12, fontweight='bold', color='black')
                
                ax.annotate(f'{worst_round_data["mean"]:.2f}', 
                           (worst_round_data['round'], worst_round_data['mean']),
                           textcoords="offset points", xytext=(0,15), ha='center',
                           fontsize=12, fontweight='bold', color='black')
                
                label = 'Best Consensus' if round_num == sorted(df['round'].unique())[0] else None
                ax.plot(best_round_data['round'], best_round_data['mean'], 
                       marker='*', markersize=15, color='gold', 
                       markeredgecolor='black', markeredgewidth=2, 
                       label=label, zorder=10)
        
        ax.set_xlabel('Discussion rounds', fontsize=16, fontweight='bold')
        ax.set_ylabel('Vote Entropy (bits)', fontsize=16, fontweight='bold') 
        ax.tick_params(axis='both', labelsize=14)
        
        rounds = sorted(df['round'].unique())
        ax.set_xticks(rounds)
        ax.set_xticklabels([r + 1 for r in rounds])  
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        ax.legend(loc='lower left', fontsize=12, frameon=True, fancybox=True, shadow=True,
                 framealpha=1.0, facecolor='white', edgecolor='black')
        
        ax.set_ylim(0.1, 0.6)
        
        if panel_label:
            ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes, 
                   fontsize=25, fontweight='bold', va='bottom', ha='left')
        
        return ax
        
    except FileNotFoundError:
        ax.text(0.5, 0.5, 'Vote Convergence Analysis\n(Data not available)', 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        if panel_label:
            ax.text(-0.1, 1.05, panel_label, transform=ax.transAxes, 
                   fontsize=16, fontweight='bold', va='bottom', ha='right')
        
        return ax