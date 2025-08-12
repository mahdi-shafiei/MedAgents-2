import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from plot_style import set_plot_style, get_color_scheme, get_marker_styles, get_line_styles, apply_standard_plot_formatting, get_standard_figure_size, get_standard_gridspec_params, get_background_colors, get_colors

set_plot_style()

df = pd.read_csv('main_comparison.csv')

fig = plt.figure(figsize=(26, 22))
gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1],  width_ratios=[2, 1, 1],
                      hspace=0.15, wspace=0.6)

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

models = ['o3-mini', 'gpt-4o', 'gpt-4o-mini']
methods = ['CoT', 'CoT-SC', 'MedPrompt', 'MultiPersona', 'MedAgents', 'AFlow', 'EBAgents', 'Few-shot', 'MDAgents', 'SPO', 'Self-refine',  'MedRAG', 'Zero-shot']
df['dataset'] = df['dataset'].map(dataset_mapping)
datasets = list(df['dataset'].unique())

ax1 = fig.add_subplot(gs[0, :])

method_stats = {}
for method in methods:
    method_df = df[df['method'] == method]
    if not method_df.empty:
        model_accuracies = {}
        model_times = {}
        for model in models:
            model_data = method_df[method_df['model'] == model]
            if not model_data.empty:
                model_accuracies[model] = {
                    'mean': model_data['accuracy'].mean(),
                    'std': model_data['accuracy'].std(),
                    'values': model_data['accuracy'].values
                }
                model_times[model] = {
                    'mean': model_data['avg_time'].mean(),
                    'std': model_data['avg_time'].std(),
                    'values': model_data['avg_time'].values
                }
        method_stats[method] = {'accuracy': model_accuracies, 'time': model_times}

overall_means = []
for method in methods:
    if method in method_stats:
        all_values = []
        for model in models:
            if model in method_stats[method]['accuracy']:
                all_values.extend(method_stats[method]['accuracy'][model]['values'])
        if all_values:
            overall_means.append((method, np.mean(all_values)))

overall_means.sort(key=lambda x: x[1], reverse=True)
methods_sorted = [item[0] for item in overall_means]

bar_width = 0.3
x_positions = np.arange(len(methods_sorted))

alphas = [0.9, 0.7, 0.5]

for i, model in enumerate(models):
    model_means = []
    model_stds = []
    model_values_list = []
    
    for method in methods_sorted:
        if method in method_stats and model in method_stats[method]['accuracy']:
            model_means.append(method_stats[method]['accuracy'][model]['mean'])
            model_stds.append(method_stats[method]['accuracy'][model]['std'])
            model_values_list.append(method_stats[method]['accuracy'][model]['values'])
        else:
            model_means.append(0)
            model_stds.append(0)
            model_values_list.append([])
    
    x_pos = x_positions + (i - 0.5) * bar_width
    
    bars = ax1.bar(x_pos, model_means, bar_width,
                   color=[method_colors[method] for method in methods_sorted],
                   alpha=alphas[i],
                   edgecolor='black', linewidth=1.5,
                   label=f'{model}')
    
    ax1.errorbar(x_pos, model_means, yerr=model_stds, fmt='none',
                 ecolor='black', capsize=4, capthick=2, linewidth=2, alpha=0.8)
    
    for j, (x, mean, values) in enumerate(zip(x_pos, model_means, model_values_list)):
        if len(values) > 0:
            for value in values:
                ax1.plot(x, value, 'o', color='darkred' if i == 0 else 'darkblue', 
                        markersize=4, alpha=0.7, markeredgecolor='black', markeredgewidth=0.5)

ax1_twin = ax1.twinx()

for i, model in enumerate(models):
    model_time_means = []
    model_time_stds = []
    
    for method in methods_sorted:
        if method in method_stats and model in method_stats[method]['time']:
            model_time_means.append(method_stats[method]['time'][model]['mean'])
            model_time_stds.append(method_stats[method]['time'][model]['std'])
        else:
            model_time_means.append(0)
            model_time_stds.append(0)
    
    line_color = 'navy' if i == 0 else 'maroon'
    line_style = '-' if i == 0 else '--'
    marker_style = 'D' if i == 0 else '^'
    
    ax1_twin.plot(x_positions, model_time_means, color=line_color, linestyle=line_style,
                  marker=marker_style, markersize=10, linewidth=4, alpha=0.9,
                  label=f'{model} (Time)', markeredgecolor='white', markeredgewidth=1)
    
    ax1_twin.errorbar(x_positions, model_time_means, yerr=model_time_stds, fmt='none',
                      ecolor=line_color, capsize=3, capthick=2, linewidth=2, alpha=0.7)

best_method_idx = 0
best_method = methods_sorted[0]
best_x = x_positions[best_method_idx]
best_y = max([method_stats[best_method]['accuracy'][model]['mean'] for model in models if model in method_stats[best_method]['accuracy']])

ax1.plot(best_x, best_y + 3, marker='*', markersize=25, color='gold', 
         markeredgecolor='black', markeredgewidth=3, zorder=15, label='Best Overall')

ax1.set_ylim(8, 48)
ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=18)
ax1_twin.set_ylabel('Execution Time (seconds)', fontweight='bold', fontsize=18)
ax1.set_xlabel('Methods (Ranked by Performance)', fontweight='bold', fontsize=18)

ax1.set_xticks(x_positions)
ax1.set_xticklabels(methods_sorted, fontsize=15)
ax1.tick_params(axis='both', labelsize=15)
ax1_twin.tick_params(axis='both', labelsize=15)
ax1.grid(True, alpha=0.4, axis='y', linewidth=0.8, linestyle=':')
apply_standard_plot_formatting(ax1, 'a', background_color=background_colors['white'], fontsize=25)

method_legend_elements = []
for method in methods_sorted[:6]:
    method_legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=method_colors[method], alpha=0.8, 
                                              edgecolor='black', linewidth=1.5, label=method))

model_legend_elements = []
for i, model in enumerate(models):
    model_legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=alphas[i],
                                             edgecolor='black', linewidth=1.5, label=f'{model} (Accuracy)'))

time_legend_elements = []
for i, model in enumerate(models):
    line_color = 'navy' if i == 0 else 'maroon'
    line_style = '-' if i == 0 else '--'
    marker_style = 'D' if i == 0 else '^'
    time_legend_elements.append(plt.Line2D([0], [0], color=line_color, linestyle=line_style,
                                         marker=marker_style, markersize=10, linewidth=4,
                                         label=f'{model} (Time)', markeredgecolor='white', markeredgewidth=1))

special_legend_elements = [plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                                    markeredgecolor='black', markeredgewidth=3, markersize=25,
                                    label='Best Overall', linestyle='None')]

combined_legend_elements = model_legend_elements + time_legend_elements + special_legend_elements

ax1.legend(handles=combined_legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
          title='Legend', fontsize=13, ncol=1,
          frameon=True, fancybox=True, shadow=True, framealpha=0.95, 
          facecolor='white', edgecolor='black', title_fontsize=15)

ax2 = fig.add_subplot(gs[1, 1:])

dataset_performance = df.groupby(['dataset', 'method']).agg({
    'accuracy': 'mean',
    'avg_time': 'mean',
    'avg_cost': 'mean'
}).reset_index()

datasets_sorted = sorted(datasets)

x_positions_heatmap = {method: i for i, method in enumerate(methods)}
y_positions_heatmap = {dataset: i for i, dataset in enumerate(datasets_sorted)}

for _, row in dataset_performance.iterrows():
    if row['method'] in methods:
        x = x_positions_heatmap[row['method']]
        y = y_positions_heatmap[row['dataset']]
        
        bubble_size = (row['accuracy'] / 45) * 1200
        bubble_alpha = min(1.0, max(0.6, row['accuracy'] / 45))
        
        time_normalized = min(1.0, row['avg_time'] / 80)
        edge_width = 2 + time_normalized * 4
        
        cost_normalized = min(1.0, (row['avg_cost'] * 100) / 60)
        edge_alpha = 0.4 + cost_normalized * 0.6
        
        bubble = ax2.scatter(x, y, 
                           s=bubble_size,
                           c=method_colors[row['method']], 
                           alpha=bubble_alpha,
                           edgecolors='black',
                           linewidths=edge_width,
                           zorder=5)
        
        if row['accuracy'] > 0:
            text_color = 'white' if row['accuracy'] > 25 else 'black'
            ax2.annotate(f"{row['accuracy']:.1f}%", 
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

for i, method in enumerate(methods):
    ax2.axvline(x=i, color='lightgray', alpha=0.5, linewidth=1, zorder=1)

for i, dataset in enumerate(datasets_sorted):
    ax2.axhline(y=i, color='lightgray', alpha=0.5, linewidth=1, zorder=1)

ax2.set_xlim(-0.5, len(methods) - 0.5)
ax2.set_ylim(-0.5, len(datasets_sorted) - 0.5)

ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=15)
ax2.set_yticks(range(len(datasets_sorted)))
ax2.set_yticklabels(datasets_sorted, fontsize=15, rotation=45)

ax2.set_xlabel('Methods', fontweight='bold', fontsize=18)
ax2.set_ylabel('Medical Datasets', fontweight='bold', fontsize=18)
ax2.tick_params(axis='both', labelsize=15)

legend_elements = []
for size, label in [(300, '15%'), (600, '30%'), (1000, '45%')]:
    legend_elements.append(plt.scatter([], [], s=size, c='gray', alpha=0.7, 
                                     edgecolors='black', linewidths=2, label=f'{label} Accuracy'))

legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Low Time/Cost'))
legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=6, label='High Time/Cost'))

ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.0, 0.5), 
          title='Bubble Size: Accuracy\nEdge Width: Time & Cost', fontsize=13, title_fontsize=15,
          frameon=True, fancybox=True, shadow=True, framealpha=0.95,
          facecolor='white', edgecolor='black')

apply_standard_plot_formatting(ax2, 'c', background_color=background_colors['white'], fontsize=25)

ax4 = fig.add_subplot(gs[1, 0])

avg_metrics = df.groupby('method').agg({
    'accuracy': 'mean',
    'avg_cost': lambda x: x.mean() * 100,
    'avg_time': 'mean'
}).reset_index()

for _, row in avg_metrics.iterrows():
    if row['method'] == 'EBAgents':
        ax4.scatter(row['avg_cost'], row['accuracy'], 
                   c=method_colors[row['method']], 
                   s=500, alpha=0.9, 
                   edgecolors='black', linewidth=3,
                   label=row['method'], marker='*', zorder=10)
    else:
        ax4.scatter(row['avg_cost'], row['accuracy'], 
                   c=method_colors[row['method']], 
                   s=250, alpha=0.8, 
                   edgecolors='black', linewidth=2,
                   label=row['method'], zorder=5)

pareto_indices = []
sorted_indices = np.argsort(avg_metrics['avg_cost'])
max_accuracy_so_far = -1
for i in sorted_indices:
    if avg_metrics.iloc[i]['accuracy'] > max_accuracy_so_far:
        pareto_indices.append(i)
        max_accuracy_so_far = avg_metrics.iloc[i]['accuracy']

if len(pareto_indices) > 1:
    pareto_data = avg_metrics.iloc[pareto_indices].sort_values('avg_cost')
    ax4.plot(pareto_data['avg_cost'], pareto_data['accuracy'], 'k--', alpha=0.8, linewidth=3, 
            label='Pareto Frontier', zorder=8)

for _, row in avg_metrics.iterrows():
    offset_x = 12 if row['method'] != 'EBAgents' else 15
    offset_y = 10 if row['method'] != 'EBAgents' else 12
    ax4.annotate(row['method'], (row['avg_cost'], row['accuracy']), 
                xytext=(offset_x, offset_y), textcoords='offset points', 
                fontsize=13, ha='left', va='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                         edgecolor='gray', linewidth=1.5))

ax4.set_xlabel('Average Cost (cents per query)', fontweight='bold', fontsize=18)
ax4.set_ylabel('Average Accuracy (%)', fontweight='bold', fontsize=18)
ax4.set_ylim(14, 34)
ax4.tick_params(axis='both', labelsize=15)
ax4.grid(True, alpha=0.4, linestyle=':', linewidth=1)
apply_standard_plot_formatting(ax4, 'b', background_color=background_colors['white'], fontsize=25)

pareto_legend = [plt.Line2D([0], [0], color='black', linestyle='--', linewidth=3, 
                           label='Pareto Frontier', alpha=0.8)]
ax4.legend(handles=pareto_legend, loc='lower right', fontsize=13, 
          frameon=True, fancybox=True, shadow=True, framealpha=0.95,
          facecolor='white', edgecolor='black')

plt.tight_layout()
plt.savefig('main_comparison.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()