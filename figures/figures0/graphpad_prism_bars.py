"""GraphPad Prism Style Bar Plots - Subfigure A"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

from plot_style import (
    set_plot_style,
    get_color_scheme,
    get_marker_styles,
    apply_standard_plot_formatting,
    get_background_colors,
)


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


def plot_graphpad_prism_bars(ax, raw, colors, panel_label='A'):
    """Plot GraphPad Prism style bar plots with stacked deltas"""
    
    method_colors = colors.get('methods', {})
    background_colors = get_background_colors()

    # Get original models and datasets
    models = ['o3-mini', 'gpt-4o', 'gpt-4o-mini']
    methods = ['CoT', 'CoT-SC', 'MedPrompt', 'MultiPersona', 'MedAgents', 'AFlow',
               'MedAgents-2', 'Few-shot', 'MDAgents', 'SPO', 'Self-refine', 'MedRAG', 'Zero-shot']

    # Normalize dataset names for display
    name_map = _dataset_name_map()
    raw['dataset'] = raw['dataset'].map(lambda x: name_map.get(x, x))

    # Arrange datasets with priority datasets in first row
    all_datasets = sorted(raw['dataset'].dropna().unique().tolist())
    priority_datasets = ['MMLU-Med', 'MedBullets', 'MedQA']
    other_datasets = [d for d in all_datasets if d not in priority_datasets]
    datasets = priority_datasets + other_datasets[:6]  # First 3 priority + 6 others for 3x3 grid
    
    # Alpha values for the 3 models (increasing transparency)
    model_alphas = [0.8, 0.6, 0.4]
    
    # Create 3x3 subplots with shared axes
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    
    # Plot each dataset
    for idx, dataset in enumerate(datasets):
        row, col = idx // 3, idx % 3
        subplot_ax = axes[row, col]
        
        # Get data and methods for this dataset
        dataset_data = raw[raw['dataset'] == dataset]
        available_methods = [m for m in methods if m in dataset_data['method'].values]
        x_positions = np.arange(len(available_methods))
        
        # Plot each method
        for method_idx, method in enumerate(available_methods):
            method_data = dataset_data[dataset_data['method'] == method]
            method_color = method_colors.get(method, '#cccccc')
            
            # Get model accuracies and standard deviations
            accuracies = []
            std_devs = []
            for model in models:
                model_data = method_data[method_data['model'] == model]
                if len(model_data) > 0:
                    runs = model_data['accuracy'].values
                    accuracies.append(np.mean(runs))
                    std_devs.append(np.std(runs) if len(runs) > 1 else 0)
                else:
                    accuracies.append(None)
                    std_devs.append(None)
            
            # Filter out None values
            valid_data = [(acc, std) for acc, std in zip(accuracies, std_devs) if acc is not None]
            if len(valid_data) > 0:
                valid_accuracies = [acc for acc, std in valid_data]
                valid_stds = [std for acc, std in valid_data]
                baseline = min(valid_accuracies)
                x_pos = x_positions[method_idx]
                
                # Plot baseline (method color with full alpha)
                subplot_ax.bar(x_pos, baseline, 0.6, color=method_color, alpha=1,
                      edgecolor='black', linewidth=0.5)
                
                # Stack deltas with model alphas
                bottom = baseline
                for model_idx, (acc, std) in enumerate(valid_data):
                    delta = acc - baseline
                    if delta > 0:
                        subplot_ax.bar(x_pos, delta, 0.6, bottom=bottom,
                              color=method_color, alpha=model_alphas[model_idx % len(model_alphas)],
                              edgecolor='black', linewidth=0.3)
                        bottom += delta
                
                # Add error bars for each model
                for model_idx, (acc, std) in enumerate(valid_data):
                    if std > 0:
                        subplot_ax.errorbar(x_pos, acc, yerr=std, 
                                   fmt='none', color='black', capsize=3, capthick=1, 
                                   linewidth=1, zorder=10)
        
        # Styling
        subplot_ax.set_title(dataset, fontweight='bold', fontsize=12, pad=10)
        
        # Only show labels on edges
        if row == 2:  # Bottom row
            subplot_ax.set_xticks(x_positions)
            subplot_ax.set_xticklabels(available_methods, rotation=90, ha='center', fontweight='bold')
        else:
            subplot_ax.set_xticks([])
            
        if col == 0:  # Left column
            pass  # Remove axis level title
        else:
            subplot_ax.tick_params(left=False, labelleft=False)
            
        # Clean styling
        subplot_ax.grid(True, alpha=0.3, linewidth=0.5, axis='y')
        subplot_ax.set_axisbelow(True)
        subplot_ax.spines['top'].set_visible(False)
        subplot_ax.spines['right'].set_visible(False)
        subplot_ax.spines['left'].set_linewidth(1.5)
        subplot_ax.spines['bottom'].set_linewidth(1.5)
        subplot_ax.set_facecolor(background_colors.get('white', '#fafafa'))
    
    # Set y-limits for all subplots
    for subplot_ax in axes.flat:
        subplot_ax.set_ylim(0, 80)
    
    # Apply standard plot formatting to first subplot
    apply_standard_plot_formatting(axes[0, 0], panel_label, background_color=background_colors['white'], fontsize=16)
    
    return fig, axes