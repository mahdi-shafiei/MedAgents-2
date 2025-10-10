"""Enhanced Orchestration Comparison with Role-play Analysis - Figure 1e"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import get_figure_1_colors, apply_medagents_style

def plot_enhanced_orchestration_comparison(ax, orchestration_df, colors, panel_label='E'):
    """Plot orchestration styles with role-play vs non-role-play comparison"""
    
    if orchestration_df.empty:
        orchestration_df = pd.DataFrame({
            'exp_name': ['group_chat_with_orchestrator', 'group_chat_voting_only', 'independent', 'one_on_one_sync',
                        'group_chat_with_orchestrator_disable_role_play', 'group_chat_voting_only_disable_role_play', 
                        'independent_disable_role_play', 'one_on_one_sync_disable_role_play'],
            'accuracy': [36.3, 33.0, 29.0, 33.7, 32.7, 29.0, 26.0, 30.0]
        })
    
    orchestration_df['has_role_play'] = ~orchestration_df['exp_name'].str.contains('disable_role_play')
    orchestration_df['base_name'] = orchestration_df['exp_name'].str.replace('_disable_role_play', '')
    
    orch_summary = orchestration_df.groupby(['base_name', 'has_role_play']).agg({
        'accuracy': ['mean', 'std']
    }).round(2)
    
    orch_summary.columns = ['accuracy_mean', 'accuracy_std']
    orch_summary = orch_summary.reset_index()
    
    order_map = {
        'group_chat_with_orchestrator': 0,
        'group_chat_voting_only': 1, 
        'independent': 2,
        'one_on_one_sync': 3
    }
    orch_summary['order'] = orch_summary['base_name'].map(order_map)
    orch_summary = orch_summary.sort_values(['order', 'has_role_play'], ascending=[True, False]).reset_index(drop=True)
    
    labels = ['Moderated\nTeam Meeting', 'Unmoderated\nGroup Chat', 'Independent\nVote', 'Individual\nMeeting']
    
    role_play_data = orch_summary[orch_summary['has_role_play'] == True]
    no_role_play_data = orch_summary[orch_summary['has_role_play'] == False]
    
    x_pos = np.arange(len(labels))
    
    ax.plot(x_pos, role_play_data['accuracy_mean'], 
            marker='o', linewidth=3, markersize=10,
            color=colors['metrics']['accuracy'], alpha=0.8,
            markeredgecolor='black', markeredgewidth=2, label='With Role-play')
    
    ax.fill_between(x_pos, 
                    role_play_data['accuracy_mean'] - role_play_data['accuracy_std'],
                    role_play_data['accuracy_mean'] + role_play_data['accuracy_std'],
                    color=colors['metrics']['accuracy'], alpha=0.2)
    
    ax.plot(x_pos, no_role_play_data['accuracy_mean'], 
            marker='s', linewidth=3, markersize=10,
            color=colors['metrics']['time'], alpha=0.8,
            markeredgecolor='black', markeredgewidth=2, label='Without Role-play')
    
    ax.fill_between(x_pos, 
                    no_role_play_data['accuracy_mean'] - no_role_play_data['accuracy_std'],
                    no_role_play_data['accuracy_mean'] + no_role_play_data['accuracy_std'],
                    color=colors['metrics']['time'], alpha=0.2)
    
    for i, (rp_acc, no_rp_acc) in enumerate(zip(role_play_data['accuracy_mean'], no_role_play_data['accuracy_mean'])):
        ax.annotate(f'{rp_acc:.1f}%', 
                   (x_pos[i], rp_acc),
                   textcoords="offset points", xytext=(0,15), ha='center',
                   fontsize=12, fontweight='bold', color='black')
        ax.annotate(f'{no_rp_acc:.1f}%', 
                   (x_pos[i], no_rp_acc),
                   textcoords="offset points", xytext=(0,-25), ha='center',
                   fontsize=12, fontweight='bold', color='black')
    
    best_rp_idx = role_play_data['accuracy_mean'].idxmax()
    best_rp_accuracy = role_play_data['accuracy_mean'].iloc[best_rp_idx]
    ax.plot(x_pos[best_rp_idx], best_rp_accuracy, marker='*', markersize=15, color='gold', 
            markeredgecolor='black', markeredgewidth=2, 
            zorder=10, label='Best performing')
    
    ax.axhline(y=best_rp_accuracy, color='gold', linestyle='--', alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Discussion Strategy', fontweight='bold', fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.2, axis='y', linewidth=0.8, linestyle='-')
    ax.set_ylim(25, 45)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=13)
    
    ax.legend(fontsize=12, loc='upper left', frameon=True, fancybox=True, shadow=True, 
              framealpha=1.0, facecolor='white', edgecolor='black')
    
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=25, fontweight='bold')
    
    return ax

def main():
    """Example usage when run as standalone script"""
    apply_medagents_style()
    colors = get_figure_1_colors()
    
    df = pd.DataFrame({
        'exp_name': ['group_chat_with_orchestrator', 'group_chat_voting_only', 'independent', 'one_on_one_sync',
                    'group_chat_with_orchestrator_disable_role_play', 'group_chat_voting_only_disable_role_play', 
                    'independent_disable_role_play', 'one_on_one_sync_disable_role_play'],
        'accuracy': [36.3, 33.0, 29.0, 33.7, 32.7, 29.0, 26.0, 30.0]
    })
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_enhanced_orchestration_comparison(ax, df, colors, panel_label='E')
    plt.tight_layout()
    plt.savefig('enhanced_orchestration_comparison_example.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()