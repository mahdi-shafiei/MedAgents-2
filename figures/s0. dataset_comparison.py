import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from plot_style import set_plot_style, get_color_scheme, get_marker_styles, get_line_styles, apply_standard_plot_formatting, get_standard_figure_size, get_standard_gridspec_params, get_background_colors, get_colors

set_plot_style()

df = pd.read_csv('main_comparison.csv')

colors = get_color_scheme('figure_0', theme='manchester_united_official')
marker_styles = get_marker_styles()
line_styles = get_line_styles()
background_colors = get_background_colors()

method_colors = colors.get('methods', {})

dataset_mapping = {
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

models = ['gpt-4o', 'gpt-4o-mini', 'o3-mini']
methods = ['CoT', 'CoT-SC', 'MedPrompt', 'MultiPersona', 'MedAgents', 'AFlow', 'EBAgents', 'Few-shot', 'MDAgents', 'SPO', 'Self-refine', 'MedRAG', 'Zero-shot']
df['dataset'] = df['dataset'].map(dataset_mapping)
datasets = list(df['dataset'].unique())

fig1 = plt.figure(figsize=(36, 48))
gs1 = fig1.add_gridspec(9, 3, hspace=0.4, wspace=0.3)

for row_idx, dataset in enumerate(datasets):
    dataset_data = df[df['dataset'] == dataset]
    
    for col_idx, model in enumerate(models):
        ax = fig1.add_subplot(gs1[row_idx, col_idx])
        
        model_data = dataset_data[dataset_data['model'] == model]
        
        if not model_data.empty:
            method_accuracies = []
            method_labels = []
            method_colors_list = []
            
            for method in methods:
                method_subset = model_data[model_data['method'] == method]
                if not method_subset.empty:
                    method_accuracies.append(method_subset['accuracy'].values)
                    method_labels.append(method)
                    method_colors_list.append(method_colors.get(method, 'gray'))
            
            if method_accuracies:
                box_plot = ax.boxplot(method_accuracies, tick_labels=method_labels, patch_artist=True,
                                    boxprops=dict(linewidth=1.5, alpha=0.8),
                                    whiskerprops=dict(linewidth=1.5),
                                    capprops=dict(linewidth=1.5),
                                    medianprops=dict(linewidth=2, color='black'))
                
                for patch, color in zip(box_plot['boxes'], method_colors_list):
                    patch.set_facecolor(color)
                    patch.set_edgecolor('black')
                
                best_method_idx = np.argmax([np.mean(acc) for acc in method_accuracies])
                best_mean = np.mean(method_accuracies[best_method_idx])
                ax.plot(best_method_idx + 1, best_mean, marker='*', markersize=12, 
                       color='gold', markeredgecolor='black', markeredgewidth=2, zorder=10)
        
        ax.set_title(f'{dataset} - {model}', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        if row_idx == 0 and col_idx == 0:
            apply_standard_plot_formatting(ax, 'a', background_color=background_colors['white'], fontsize=25)
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('performance_comparison.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

fig2 = plt.figure(figsize=(36, 48))
gs2 = fig2.add_gridspec(9, 3, hspace=0.4, wspace=0.3)

for row_idx, dataset in enumerate(datasets):
    dataset_data = df[df['dataset'] == dataset]
    
    for col_idx, model in enumerate(models):
        ax = fig2.add_subplot(gs2[row_idx, col_idx])
        
        model_data = dataset_data[dataset_data['model'] == model]
        
        if not model_data.empty:
            dataset_metrics = model_data.groupby('method').agg({
                'accuracy': 'mean',
                'avg_cost': lambda x: x.mean() * 100,
                'avg_time': 'mean'
            }).reset_index()
            
            best_method_idx = dataset_metrics['accuracy'].idxmax()
            
            for idx, row_data in dataset_metrics.iterrows():
                if idx == best_method_idx:
                    ax.scatter(row_data['avg_cost'], row_data['accuracy'], 
                             c=method_colors[row_data['method']], 
                             s=400, alpha=0.9, 
                             edgecolors='black', linewidth=2,
                             marker='*', zorder=10)
                else:
                    ax.scatter(row_data['avg_cost'], row_data['accuracy'], 
                             c=method_colors[row_data['method']], 
                             s=200, alpha=0.8, 
                             edgecolors='black', linewidth=1.5,
                             zorder=5)
            
            dataset_pareto_indices = []
            dataset_sorted_indices = np.argsort(dataset_metrics['avg_cost'])
            dataset_max_accuracy_so_far = -1
            for i in dataset_sorted_indices:
                if dataset_metrics.iloc[i]['accuracy'] > dataset_max_accuracy_so_far:
                    dataset_pareto_indices.append(i)
                    dataset_max_accuracy_so_far = dataset_metrics.iloc[i]['accuracy']
            
            if len(dataset_pareto_indices) > 1:
                dataset_pareto_data = dataset_metrics.iloc[dataset_pareto_indices].sort_values('avg_cost')
                ax.plot(dataset_pareto_data['avg_cost'], dataset_pareto_data['accuracy'], 
                       'k--', alpha=0.8, linewidth=2, zorder=8)
            
            for _, row_data in dataset_metrics.iterrows():
                ax.annotate(row_data['method'], (row_data['avg_cost'], row_data['accuracy']), 
                          xytext=(5, 5), textcoords='offset points', 
                          fontsize=8, ha='left', va='bottom', fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, 
                                   edgecolor='gray', linewidth=0.5))
        
        ax.set_xlabel('Cost (cents per query)', fontweight='bold', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=10)
        ax.set_title(f'{dataset} - {model}', fontweight='bold', fontsize=12)
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        if row_idx == 0 and col_idx == 0:
            apply_standard_plot_formatting(ax, 'a', background_color=background_colors['white'], fontsize=25)
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('pareto_analysis.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()