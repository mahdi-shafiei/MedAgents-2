"""Enhanced Orchestration Comparison with Dual Y-axis - Figure 1e"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import get_figure_1_colors, apply_medagents_style

def plot_enhanced_orchestration_comparison(ax, orchestration_df, colors, panel_label='E'):
    """Plot orchestration styles with both accuracy and time metrics"""
    
    if orchestration_df.empty:
        # Create default data if empty
        orchestration_df = pd.DataFrame({
            'exp_name': ['group_chat_with_orchestrator', 'group_chat_voting_only', 'independent', 'one_on_one_sync'],
            'accuracy': [35, 32, 28, 30],
            'avg_time': [70, 65, 45, 55]
        })
    
    # Process data
    orch_summary = orchestration_df.groupby('exp_name').agg({
        'accuracy': ['mean', 'std'],
        'avg_time': ['mean', 'std']
    }).round(2)
    
    orch_summary.columns = ['accuracy_mean', 'accuracy_std', 'time_mean', 'time_std']
    orch_summary = orch_summary.reset_index()
    
    # Sort by predefined order
    order_map = {
        'group_chat_with_orchestrator': 0,
        'group_chat_voting_only': 1, 
        'independent': 2,
        'one_on_one_sync': 3
    }
    orch_summary['order'] = orch_summary['exp_name'].map(order_map)
    orch_summary = orch_summary.sort_values('order').reset_index(drop=True)
    
    # Define labels and colors
    labels = ['Moderated\nTeam Meeting', 'Unmoderated\nGroup Chat', 'Individual\nMeeting', 'Independent\nVote']
    orchestration_colors = [
        colors['orchestration']['group_chat_with_orchestrator'],
        colors['orchestration']['group_chat_voting_only'],
        colors['orchestration']['independent'],
        colors['orchestration']['one_on_one_sync']
    ]
    
    x_pos = np.arange(len(orch_summary))
    
    # Plot accuracy line
    ax.plot(x_pos, orch_summary['accuracy_mean'], 
            marker='o', linewidth=3, markersize=10,
            color=colors['metrics']['accuracy'], alpha=0.8,
            markeredgecolor='black', markeredgewidth=2, label='Accuracy')
    
    # Add accuracy confidence intervals
    ax.fill_between(x_pos, 
                    orch_summary['accuracy_mean'] - orch_summary['accuracy_std'],
                    orch_summary['accuracy_mean'] + orch_summary['accuracy_std'],
                    color=colors['metrics']['accuracy'], alpha=0.2)
    
    # Add accuracy value annotations
    for i, acc in enumerate(orch_summary['accuracy_mean']):
        ax.annotate(f'{acc:.1f}%', 
                   (x_pos[i], acc),
                   textcoords="offset points", xytext=(0,15), ha='center',
                   fontsize=12, fontweight='bold', color='black')
    
    # Highlight best performing configuration
    best_idx = orch_summary['accuracy_mean'].idxmax()
    best_accuracy = orch_summary['accuracy_mean'].iloc[best_idx]
    ax.plot(x_pos[best_idx], best_accuracy, marker='*', markersize=15, color='gold', 
            markeredgecolor='black', markeredgewidth=2, 
            zorder=10, label='Best performing')
    
    ax.axhline(y=best_accuracy, color='gold', linestyle='--', alpha=0.7, linewidth=2)
    
    # Create second y-axis for time
    ax_time = ax.twinx()
    
    # Plot time line
    time_line = ax_time.plot(x_pos, orch_summary['time_mean'], 
                            marker='s', linewidth=2.5, markersize=8,
                            color=colors['metrics']['time'], alpha=0.7,
                            markeredgecolor='black', markeredgewidth=1.5, 
                            linestyle='--', label='Processing Time')
    
    # Add time confidence intervals
    ax_time.fill_between(x_pos, 
                        orch_summary['time_mean'] - orch_summary['time_std'],
                        orch_summary['time_mean'] + orch_summary['time_std'],
                        color=colors['metrics']['time'], alpha=0.15)
    
    # Styling for main axis (accuracy)
    ax.set_xlabel('Discussion Strategy', fontweight='bold', fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.2, axis='y', linewidth=0.8, linestyle='-')
    ax.set_ylim(20, 45)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=13)
    
    # Styling for time axis
    ax_time.set_ylabel('Processing Time (s)', fontweight='bold', fontsize=16, color='black')
    ax_time.tick_params(axis='y', labelcolor='black', labelsize=13)
    
    # Add legends
    ax.legend(fontsize=12, loc='upper left', frameon=True, fancybox=True, shadow=True, 
              framealpha=1.0, facecolor='white', edgecolor='black')
    ax_time.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=True, 
                   framealpha=1.0, facecolor='white', edgecolor='black')
    
    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=25, fontweight='bold')
    
    return ax, ax_time

def main():
    """Example usage when run as standalone script"""
    apply_medagents_style()
    colors = get_figure_1_colors()
    
    # Create sample data
    df = pd.DataFrame({
        'exp_name': ['group_chat_with_orchestrator', 'group_chat_voting_only', 'independent', 'one_on_one_sync'],
        'accuracy': [35, 32, 28, 30],
        'avg_time': [70, 65, 45, 55]
    })
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_enhanced_orchestration_comparison(ax, df, colors, panel_label='E')
    plt.tight_layout()
    plt.savefig('enhanced_orchestration_comparison_example.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()