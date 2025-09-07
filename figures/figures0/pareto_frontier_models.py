"""Pareto Frontier Plot by Models - Subfigure B"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot_style import apply_standard_plot_formatting, get_background_colors


def _dataset_name_map():
    return {
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


def plot_pareto_frontier_models(raw, colors, panel_label='B'):
    """Create 3x1 Pareto frontier plots for each model"""
    
    method_colors = colors.get('methods', {})
    background_colors = get_background_colors()
    
    # Normalize dataset names for display
    name_map = _dataset_name_map()
    raw['dataset'] = raw['dataset'].map(lambda x: name_map.get(x, x))
    
    # Get models
    models = ['o3-mini', 'gpt-4o', 'gpt-4o-mini']
    
    # Create 3x1 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    for model_idx, model in enumerate(models):
        ax = axes[model_idx]
        
        # Filter data for this model
        model_data = raw[raw['model'] == model]
        
        # Calculate average metrics per method for this model
        avg_metrics = model_data.groupby('method').agg({
            'accuracy': 'mean',
            'avg_cost': lambda x: x.mean() * 100,  # Convert to cents
            'avg_time': 'mean'
        }).reset_index()
        
        if len(avg_metrics) == 0:
            continue
            
        # Plot scatter points
        for _, row in avg_metrics.iterrows():
            if row['method'] == 'MedAgents-2':
                ax.scatter(row['avg_cost'], row['accuracy'], 
                          c=method_colors.get(row['method'], 'gray'), 
                          s=400, alpha=0.9, 
                          edgecolors='black', linewidth=2.5,
                          marker='*', zorder=10)
            else:
                ax.scatter(row['avg_cost'], row['accuracy'], 
                          c=method_colors.get(row['method'], 'gray'), 
                          s=180, alpha=0.8, 
                          edgecolors='black', linewidth=1.5,
                          zorder=5)

        # Calculate Pareto frontier
        pareto_indices = []
        sorted_indices = np.argsort(avg_metrics['avg_cost'])
        max_accuracy_so_far = -1
        for i in sorted_indices:
            if avg_metrics.iloc[i]['accuracy'] > max_accuracy_so_far:
                pareto_indices.append(i)
                max_accuracy_so_far = avg_metrics.iloc[i]['accuracy']

        if len(pareto_indices) > 1:
            pareto_data = avg_metrics.iloc[pareto_indices].sort_values('avg_cost')
            ax.plot(pareto_data['avg_cost'], pareto_data['accuracy'], 'k--', alpha=0.8, linewidth=2.5, 
                   zorder=8)

        # Add method annotations for key methods
        key_methods = ['MedAgents-2', 'CoT', 'Few-shot', 'Zero-shot', 'MedRAG']
        for _, row in avg_metrics.iterrows():
            if row['method'] in key_methods:
                offset_x = 8 if row['method'] != 'MedAgents-2' else 10
                offset_y = 6 if row['method'] != 'MedAgents-2' else 8
                ax.annotate(row['method'], (row['avg_cost'], row['accuracy']), 
                           xytext=(offset_x, offset_y), textcoords='offset points', 
                           fontsize=9, ha='left', va='bottom', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                                    edgecolor='gray', linewidth=1))

        # Styling
        ax.set_title(model, fontweight='bold', fontsize=14, pad=15)
        ax.set_xlabel('Average Cost (cents per query)', fontweight='bold', fontsize=12)
        if model_idx == 0:  # Only leftmost subplot gets y-label
            ax.set_ylabel('Average Accuracy (%)', fontweight='bold', fontsize=12)
        
        # Set appropriate x-limits based on model
        if model == 'gpt-4o-mini':
            ax.set_xlim(-0.5, 5.0)
        elif model == 'gpt-4o':
            ax.set_xlim(-0.5, 20.0)
        elif model == 'o3-mini':
            ax.set_xlim(-0.5, 10.0)
        
        ax.set_ylim(10, 75)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        ax.set_facecolor(background_colors.get('white', '#fafafa'))
        
        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Apply panel label to first subplot
        if model_idx == 0:
            apply_standard_plot_formatting(ax, panel_label, background_color=background_colors['white'], fontsize=16)
    
    # Add Pareto frontier legend to the last subplot
    pareto_legend = [plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2.5, 
                               label='Pareto Frontier', alpha=0.8)]
    axes[-1].legend(handles=pareto_legend, loc='lower right', fontsize=10, 
                   frameon=True, fancybox=True, shadow=True, framealpha=0.95,
                   facecolor='white', edgecolor='black')
    
    return fig, axes