"""Role Play Comparison Plot - Figure 1c"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import get_manchester_colors, get_figure_1_colors, apply_medagents_style

def plot_role_play_comparison(ax, role_play_df, colors, panel_label='C'):
    """Plot role play vs no role play comparison"""
    
    if not role_play_df.empty:
        role_play_summary = role_play_df.groupby('exp_name').agg({
            'accuracy': ['mean', 'std']
        }).round(2)
        
        role_play_summary.columns = ['accuracy_mean', 'accuracy_std']
        role_play_summary = role_play_summary.reset_index()
        
        # Map experiment names to readable labels
        role_play_summary['label'] = role_play_summary['exp_name'].replace({
            'enable_role_play': 'With Role Play',
            'disable_role_play': 'Without Role Play'
        })
        
        colors_list = [colors['role_play']['enable_role_play'], colors['role_play']['disable_role_play']]
        
        bars = ax.bar(range(len(role_play_summary)), role_play_summary['accuracy_mean'], 
                     color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Error bars
        ax.errorbar(range(len(role_play_summary)), role_play_summary['accuracy_mean'], 
                   yerr=role_play_summary['accuracy_std'], fmt='none',
                   ecolor='black', capsize=5, capthick=2, linewidth=2)
        
        # Add value annotations
        for i, (bar, row) in enumerate(zip(bars, role_play_summary.itertuples())):
            height = bar.get_height()
            ax.annotate(f'{row.accuracy_mean:.1f}%', 
                       (bar.get_x() + bar.get_width()/2., height + 1),
                       ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_xticks(range(len(role_play_summary)))
        ax.set_xticklabels(role_play_summary['label'], fontsize=14)
    
    # Styling
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax.set_ylim(25, 40)
    
    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=25, fontweight='bold')
    
    return ax

def main():
    """Example usage when run as standalone script"""
    apply_medagents_style()
    colors = get_figure_1_colors()
    
    # Load real data
    try:
        df = pd.read_csv('../role_play.csv')
        print(f"Loaded {len(df)} role play records from real data")
    except FileNotFoundError:
        print("Warning: role_play.csv not found, creating sample data")
        df = pd.DataFrame({
            'exp_name': ['enable_role_play', 'disable_role_play'],
            'accuracy': [35, 30],
            'avg_time': [60, 45]
        })
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_role_play_comparison(ax, df, colors, panel_label='C')
    plt.tight_layout()
    plt.savefig('role_play_comparison_example.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()