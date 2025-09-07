"""Agent Scaling Analysis - Figure 1a"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import get_figure_1_colors, apply_medagents_style

def plot_agent_scaling_analysis(ax, df, colors, panel_label='A'):
    """Plot how performance changes with number of agents"""
    
    # Configuration
    AGENT_MAPPING = {1: 0, 2: 1, 3: 2, 5: 3}
    
    # Parse experiment names
    def parse_exp_name(exp_name):
        parts = exp_name.split('_')
        n_agents = int(parts[0])
        n_rounds = int(parts[2])
        has_search = 'with_search' in exp_name
        return n_agents, n_rounds, has_search

    df[['n_agents', 'n_rounds', 'has_search']] = df['exp_name'].apply(
        lambda x: pd.Series(parse_exp_name(x))
    )

    # Calculate summary statistics
    summary_df = df.groupby(['n_agents', 'n_rounds', 'has_search']).agg({
        'accuracy': ['mean', 'std'],
        'avg_time': ['mean', 'std'],
        'avg_cost': 'mean'
    }).round(2)

    summary_df.columns = ['accuracy_mean', 'accuracy_std', 'time_mean', 'time_std', 'cost_mean']
    summary_df = summary_df.reset_index()
    summary_df['n_agents_mapped'] = summary_df['n_agents'].map(AGENT_MAPPING)

    # Plot with search (1 round)
    agent_data_search = summary_df[(summary_df['n_rounds'] == 1) & (summary_df['has_search'] == True)]
    if not agent_data_search.empty:
        ax.plot(agent_data_search['n_agents_mapped'], agent_data_search['accuracy_mean'], 
               marker='o', linewidth=3, markersize=10,
               color=colors['metrics']['accuracy'], alpha=0.8,
               markeredgecolor='black', markeredgewidth=2, label='With search')
        
        ax.fill_between(agent_data_search['n_agents_mapped'], 
                       agent_data_search['accuracy_mean'] - agent_data_search['accuracy_std'],
                       agent_data_search['accuracy_mean'] + agent_data_search['accuracy_std'],
                       color=colors['metrics']['accuracy'], alpha=0.2)
        
        # Add value annotations
        for i, row in agent_data_search.iterrows():
            ax.annotate(f'{row["accuracy_mean"]:.1f}%', 
                       (row['n_agents_mapped'], row['accuracy_mean']),
                       textcoords="offset points", xytext=(0,15), ha='center',
                       fontsize=12, fontweight='bold', color='black')

    # Plot without search (1 round)
    agent_data_no_search = summary_df[(summary_df['n_rounds'] == 1) & (summary_df['has_search'] == False)]
    if not agent_data_no_search.empty:
        ax.plot(agent_data_no_search['n_agents_mapped'], agent_data_no_search['accuracy_mean'], 
               marker='s', linewidth=3, markersize=10,
               color=colors['metrics']['time'], alpha=0.8,
               markeredgecolor='black', markeredgewidth=2, label='No search')
        
        ax.fill_between(agent_data_no_search['n_agents_mapped'], 
                       agent_data_no_search['accuracy_mean'] - agent_data_no_search['accuracy_std'],
                       agent_data_no_search['accuracy_mean'] + agent_data_no_search['accuracy_std'],
                       color=colors['metrics']['time'], alpha=0.2)
        
        # Add value annotations
        for i, row in agent_data_no_search.iterrows():
            ax.annotate(f'{row["accuracy_mean"]:.1f}%', 
                       (row['n_agents_mapped'], row['accuracy_mean']),
                       textcoords="offset points", xytext=(0,-20), ha='center',
                       fontsize=12, fontweight='bold', color='black')

    # Highlight best performing configuration
    if not summary_df[(summary_df['n_rounds'] == 1)].empty:
        best_accuracy_idx = summary_df[(summary_df['n_rounds'] == 1)]['accuracy_mean'].idxmax()
        best_point = summary_df.loc[best_accuracy_idx]
        best_x = AGENT_MAPPING[best_point['n_agents']]
        best_y = best_point['accuracy_mean']
        ax.plot(best_x, best_y, marker='*', markersize=15, color='gold', 
               markeredgecolor='black', markeredgewidth=2, label='Best performing')

        ax.axhline(y=best_y, color='gold', linestyle='--', alpha=0.7, linewidth=2)

    # Styling
    ax.set_xlabel('Number of agents', fontweight='bold', fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=12, loc='upper left', frameon=True, fancybox=True, shadow=True, 
             framealpha=1.0, facecolor='white', edgecolor='black')
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(20, 45)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([1, 2, 3, 5])
    
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
        df = pd.read_csv('../agent_configuration.csv')
        print(f"Loaded {len(df)} agent configuration records from real data")
    except FileNotFoundError:
        print("Warning: agent_configuration.csv not found, creating sample data")
        df = pd.DataFrame({
            'exp_name': ['1_agent_1_round_no_search', '2_agents_1_round_no_search', '3_agents_1_round_no_search',
                        '1_agent_1_round_with_search', '2_agents_1_round_with_search', '3_agents_1_round_with_search'],
            'accuracy': [20, 25, 28, 30, 33, 35],
            'avg_time': [15, 25, 35, 45, 55, 65],
            'avg_cost': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        })
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_agent_scaling_analysis(ax, df, colors, panel_label='A')
    plt.tight_layout()
    plt.savefig('agent_scaling_analysis_example.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()