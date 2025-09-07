"""Orchestration Comparison Plot - Figure 1d"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import get_manchester_colors, get_figure_1_colors, apply_medagents_style

def plot_orchestration_comparison(ax, orchestration_df, colors, panel_label='D'):
    """Plot different orchestration styles"""
    
    if not orchestration_df.empty:
        orch_summary = orchestration_df.groupby('exp_name').agg({
            'accuracy': ['mean', 'std']
        }).round(2)
        
        orch_summary.columns = ['accuracy_mean', 'accuracy_std']
        orch_summary = orch_summary.reset_index()
        
        # Get colors for orchestration styles
        colors_list = [colors['orchestration'].get(exp_name, colors['metrics']['accuracy']) 
                      for exp_name in orch_summary['exp_name']]
        
        bars = ax.bar(range(len(orch_summary)), orch_summary['accuracy_mean'], 
                     color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Error bars
        ax.errorbar(range(len(orch_summary)), orch_summary['accuracy_mean'], 
                   yerr=orch_summary['accuracy_std'], fmt='none',
                   ecolor='black', capsize=5, capthick=2, linewidth=2)
        
        # Add value annotations
        for i, (bar, row) in enumerate(zip(bars, orch_summary.itertuples())):
            height = bar.get_height()
            ax.annotate(f'{row.accuracy_mean:.1f}%', 
                       (bar.get_x() + bar.get_width()/2., height + 1),
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Clean up labels
        labels = [name.replace('_', ' ').title() for name in orch_summary['exp_name']]
        ax.set_xticks(range(len(orch_summary)))
        ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
    
    # Styling
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax.set_ylim(25, 40)
    
    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=20, fontweight='bold')
    
    return ax

def main():
    """Example usage when run as standalone script"""
    apply_medagents_style()
    colors = get_figure_1_colors()
    
    # Load real data
    try:
        df = pd.read_csv('../orchestration_style.csv')
        print(f"Loaded {len(df)} orchestration records from real data")
    except FileNotFoundError:
        print("Warning: orchestration_style.csv not found, creating sample data")
        df = pd.DataFrame({
            'exp_name': ['group_chat_with_orchestrator', 'group_chat_voting_only', 'independent', 'one_on_one_sync'],
            'accuracy': [35, 32, 28, 30],
            'avg_time': [70, 65, 45, 55]
        })
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_orchestration_comparison(ax, df, colors, panel_label='D')
    plt.tight_layout()
    plt.savefig('orchestration_comparison_example.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()