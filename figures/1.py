"""Figure 1: Agent Configuration Analysis - Modular with Subfigures"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add figure1 directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'figure1'))

from figure1.plot_utils import apply_medagents_style, get_figure_1_colors
from figure1.agent_scaling_analysis import plot_agent_scaling_analysis
from figure1.rounds_analysis import plot_rounds_analysis
from figure1.role_play_comparison import plot_role_play_comparison
from figure1.orchestration_comparison import plot_orchestration_comparison
from figure1.enhanced_orchestration_comparison import plot_enhanced_orchestration_comparison
from figure1.expert_specialty_distribution import plot_expert_specialty_distribution
from figure1.triage_display import plot_triage_pdf
from figure1.discussion_patterns_display import plot_discussion_patterns_pdf
from figure1.vote_convergence_plot import plot_vote_convergence

def create_figure_1():
    """Create Figure 1: Agent Configuration Analysis"""
    
    # Load original MedAgents style
    apply_medagents_style()
    
    # Get color scheme
    colors = get_figure_1_colors()
    
    # Load real data
    try:
        df = pd.read_csv('agent_configuration.csv')
        print(f"Loaded {len(df)} agent configuration records from real data")
    except FileNotFoundError:
        print("Warning: agent_configuration.csv not found, creating sample data")
        df = pd.DataFrame({
            'exp_name': ['1_agent_1_round_no_search', '2_agents_1_round_no_search', '3_agents_1_round_no_search',
                        '1_agent_1_round_with_search', '2_agents_1_round_with_search', '3_agents_1_round_with_search',
                        '3_agents_2_rounds_with_search', '3_agents_3_rounds_with_search'],
            'accuracy': [20, 25, 28, 30, 33, 35, 36, 34],
            'avg_time': [15, 25, 35, 45, 55, 65, 85, 105],
            'avg_cost': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.55]
        })
    
    try:
        role_play_df = pd.read_csv('role_play.csv')
        print(f"Loaded {len(role_play_df)} role play records from real data")
    except FileNotFoundError:
        print("Warning: role_play.csv not found, creating sample data")
        role_play_df = pd.DataFrame({
            'exp_name': ['enable_role_play', 'disable_role_play'],
            'accuracy': [35, 30],
            'avg_time': [60, 45]
        })
    
    try:
        orchestration_df = pd.read_csv('orchestration_style.csv')
        print(f"Loaded {len(orchestration_df)} orchestration records from real data")
    except FileNotFoundError:
        print("Warning: orchestration_style.csv not found, creating sample data")
        orchestration_df = pd.DataFrame({
            'exp_name': ['group_chat_with_orchestrator', 'group_chat_voting_only', 'independent', 'one_on_one_sync'],
            'accuracy': [35, 32, 28, 30],
            'avg_time': [70, 65, 45, 55]
        })
    
    try:
        expert_profiles_df = pd.read_csv('expert_profiles.csv')
        print(f"Loaded {len(expert_profiles_df)} expert profile records from real data")
    except FileNotFoundError:
        print("Warning: expert_profiles.csv not found, creating sample data")
        # Create sample expert profiles data
        sample_data = []
        datasets = ['medqa', 'medmcqa', 'pubmedqa', 'mmlu']
        specialties = ['Internal Medicine', 'Cardiology', 'Emergency Medicine', 'Pediatrics', 
                      'Neurology', 'Infectious Disease', 'Endocrinology', 'Pharmacology']
        
        for dataset in datasets:
            for specialty in specialties:
                # Add multiple entries to simulate distribution
                for _ in range(np.random.randint(2, 6)):
                    sample_data.append({'dataset': dataset, 'job_title': specialty})
        
        expert_profiles_df = pd.DataFrame(sample_data)
    
    try:
        vote_entropy_df = pd.read_csv('vote_entropy_summary.csv')
        print(f"Loaded {len(vote_entropy_df)} vote entropy records from real data")
    except FileNotFoundError:
        print("Warning: vote_entropy_summary.csv not found, creating sample data")
        vote_entropy_df = pd.DataFrame({
            'exp_name': ['vote_entropy_summary'],
            'vote_entropy': [0.5]
        })
    
    # Create figure and grid - updated layout with 4 rows
    fig = plt.figure(figsize=(18, 24))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1.6, 1.4, 1.2], width_ratios=[1.3, 1.2, 1.5], 
                          hspace=0.3, wspace=0.2)
    
    # Row 1: Agent scaling analysis, Rounds analysis, Enhanced orchestration comparison
    ax1 = fig.add_subplot(gs[0, 0])
    plot_agent_scaling_analysis(ax1, df, colors, panel_label='A')
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_rounds_analysis(ax2, df, colors, panel_label='B')
    
    ax7 = fig.add_subplot(gs[0, 2])
    plot_enhanced_orchestration_comparison(ax7, orchestration_df, colors, panel_label='C')
    
    # Row 2: Vote convergence and Triage PDF
    ax8 = fig.add_subplot(gs[1, 0])
    plot_vote_convergence(ax8, vote_entropy_df, panel_label='D')
    
    ax3 = fig.add_subplot(gs[1, 1:])
    plot_triage_pdf(ax3, panel_label='E')
    
    # Row 3: Expert specialty distribution and Role play
    ax5 = fig.add_subplot(gs[2, :2])
    plot_expert_specialty_distribution(ax5, expert_profiles_df, colors, panel_label='F')
    
    ax6 = fig.add_subplot(gs[2, 2])
    plot_role_play_comparison(ax6, role_play_df, colors, panel_label='G')
    
    # Row 4: Discussion patterns PDF (spanning all columns) - moved to bottom
    ax4 = fig.add_subplot(gs[3, :])
    plot_discussion_patterns_pdf(ax4, panel_label='H')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('fig-3.agent_configuration_analysis.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

if __name__ == "__main__":
    create_figure_1()