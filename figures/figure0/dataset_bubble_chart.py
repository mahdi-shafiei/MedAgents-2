"""Dataset Bubble Chart - Figure 0c"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import get_manchester_colors, get_figure_0_colors, apply_medagents_style

def plot_dataset_bubble_chart(ax, df, colors, panel_label):
    """Create dataset performance bubble chart"""
    
    # Configuration
    DATASET_MAPPING = {
        'medbullets': 'MedBullets',
        'medexqa': 'MedExQA',
        'medmcqa': 'MedMCQA',
        'medqa': 'MedQA',
        'medxpertqa-r': 'MedXpertQA-R',
        'medxpertqa-u': 'MedXpertQA-U',
        'mmlu': 'MMLU-Med',
        'mmlu-pro': 'MMLU-Pro-Med',
        'pubmedqa': 'PubMedQA',
    }
    
    METHODS = ['CoT', 'CoT-SC', 'MedPrompt', 'MultiPersona', 'MedAgents', 'AFlow', 
               'MedAgents-2', 'Few-shot', 'MDAgents', 'SPO', 'Self-refine', 'MedRAG', 'Zero-shot']
    
    method_colors = colors.get('methods', {})
    
    # Apply dataset mapping
    df['dataset'] = df['dataset'].map(DATASET_MAPPING)
    datasets = list(df['dataset'].unique())
    
    # Calculate dataset performance
    dataset_performance = df.groupby(['dataset', 'method']).agg({
        'accuracy': 'mean',
        'avg_time': 'mean',
        'avg_cost': 'mean'
    }).reset_index()

    datasets_sorted = sorted(datasets)
    
    # Create position mappings
    x_positions_heatmap = {method: i for i, method in enumerate(METHODS)}
    y_positions_heatmap = {dataset: i for i, dataset in enumerate(datasets_sorted)}

    # Plot bubbles
    for _, row in dataset_performance.iterrows():
        if row['method'] in METHODS:
            x = x_positions_heatmap[row['method']]
            y = y_positions_heatmap[row['dataset']]
            
            # Calculate bubble properties
            bubble_size = (row['accuracy'] / 45) * 1200
            bubble_alpha = min(1.0, max(0.6, row['accuracy'] / 45))
            
            time_normalized = min(1.0, row['avg_time'] / 80)
            edge_width = 2 + time_normalized * 4
            
            cost_normalized = min(1.0, (row['avg_cost'] * 100) / 60)
            edge_alpha = 0.4 + cost_normalized * 0.6
            
            ax.scatter(x, y, 
                      s=bubble_size,
                      c=method_colors.get(row['method'], 'gray'), 
                      alpha=bubble_alpha,
                      edgecolors='black',
                      linewidths=edge_width,
                      zorder=5)
            
            # Add text annotation
            if row['accuracy'] > 0:
                text_color = 'white' if row['accuracy'] > 25 else 'black'
                ax.annotate(f"{row['accuracy']:.1f}%", 
                           (x, y), 
                           xytext=(0, 0), 
                           textcoords='offset points',
                           ha='center', 
                           va='center',
                           fontsize=11,
                           fontweight='bold',
                           color=text_color,
                           alpha=0.9,
                           zorder=10)

    # Add grid lines
    for i, method in enumerate(METHODS):
        ax.axvline(x=i, color='lightgray', alpha=0.5, linewidth=1, zorder=1)

    for i, dataset in enumerate(datasets_sorted):
        ax.axhline(y=i, color='lightgray', alpha=0.5, linewidth=1, zorder=1)

    # Styling
    ax.set_xlim(-0.5, len(METHODS) - 0.5)
    ax.set_ylim(-0.5, len(datasets_sorted) - 0.5)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=45, ha='right', fontsize=15)
    ax.set_yticks(range(len(datasets_sorted)))
    ax.set_yticklabels(datasets_sorted, fontsize=15, rotation=45)
    ax.set_xlabel('Methods', fontweight='bold', fontsize=18)
    ax.set_ylabel('Medical Datasets', fontweight='bold', fontsize=18)
    ax.tick_params(axis='both', labelsize=15)

    # Legend
    legend_elements = []
    for size, label in [(300, '15%'), (600, '30%'), (1000, '45%')]:
        legend_elements.append(plt.scatter([], [], s=size, c='gray', alpha=0.7, 
                                         edgecolors='black', linewidths=2, label=f'{label} Accuracy'))

    legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Low Time/Cost'))
    legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=6, label='High Time/Cost'))

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.0, 0.5), 
             title='Bubble Size: Accuracy\nEdge Width: Time & Cost', fontsize=13, title_fontsize=15,
             frameon=True, fancybox=True, shadow=True, framealpha=0.95,
             facecolor='white', edgecolor='black')

    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=25, fontweight='bold')
    
    return ax

def main():
    """Example usage when run as standalone script"""
    apply_medagents_style()
    colors = get_figure_0_colors()
    
    # Load real data
    try:
        df = pd.read_csv('../main_comparison.csv')
        print(f"Loaded {len(df)} records from real data")
    except FileNotFoundError:
        print("Warning: main_comparison.csv not found, creating sample data")
        methods = ['MedAgents-2', 'CoT', 'Zero-shot'] * 3
        datasets = ['medbullets', 'medqa', 'pubmedqa'] * 3
        df = pd.DataFrame({
            'method': methods,
            'dataset': datasets,
            'accuracy': [42.5, 35.2, 18.5, 38.7, 31.5, 16.2, 35.8, 28.9, 15.1],
            'avg_time': [65.3, 42.1, 15.2, 58.7, 38.2, 13.8, 61.5, 40.8, 14.2],
            'avg_cost': [0.125, 0.085, 0.023, 0.118, 0.082, 0.021, 0.121, 0.084, 0.022]
        })
    
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_dataset_bubble_chart(ax, df, colors, panel_label='C')
    plt.tight_layout()
    plt.savefig('dataset_bubble_chart_example.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()