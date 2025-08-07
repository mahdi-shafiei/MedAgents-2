import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from plot_style import set_plot_style, get_color_scheme, get_marker_styles, get_line_styles, apply_standard_plot_formatting, get_standard_figure_size, get_standard_gridspec_params, get_background_colors, get_colors

set_plot_style()

df = pd.read_csv('main_comparison.csv')
df.rename(columns={'exp_name': 'method'}, inplace=True)

fig = plt.figure(figsize=(24, 20))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1],  width_ratios=[1.5, 1, 1],
                      hspace=0.2, wspace=0.3)

colors = get_color_scheme('figure_1')
marker_styles = get_marker_styles()
line_styles = get_line_styles()
background_colors = get_background_colors()
manchester_colors = get_colors('manchester_united_official')

method_colors = {
    'EBAgents': manchester_colors['jasmine'],
    'AFlow': manchester_colors['sunset'],
    'Self-refine': manchester_colors['sandy_brown'],
    'MultiPersona': manchester_colors['persimmon'],
    'SPO': manchester_colors['barn_red'],
    'CoT': manchester_colors['penn_red'],
    'CoT-SC': manchester_colors['engineering_orange'],
    'MedAgents': manchester_colors['pink'],
    'MedPrompt': manchester_colors['snow'],
    'Few-shot': manchester_colors['gray'],
    'MDAgents': manchester_colors['jet'],
    'Zero-shot': manchester_colors['black']
}

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

models = ['gpt-4o', 'gpt-4o-mini']
methods = ['CoT', 'CoT-SC', 'MedPrompt', 'MultiPersona', 'MedAgents', 'AFlow', 'EBAgents', 'Few-shot', 'MDAgents', 'SPO', 'Self-refine', 'Zero-shot']
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

bar_width = 0.35
x_positions = np.arange(len(methods_sorted))

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
                   alpha=0.8 if i == 0 else 0.4,
                   edgecolor='black', linewidth=2,
                   label=f'{model} (Accuracy)')
    
    ax1.errorbar(x_pos, model_means, yerr=model_stds, fmt='none',
                 ecolor='black', capsize=3, capthick=2, linewidth=2)
    
    for j, (x, mean, values) in enumerate(zip(x_pos, model_means, model_values_list)):
        if len(values) > 0:
            for value in values:
                ax1.plot(x, value, 'o', color='black', markersize=3, alpha=0.6)

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
    
    line_color = 'darkblue' if i == 0 else 'darkred'
    line_style = '-' if i == 0 else '--'
    marker_style = 'o' if i == 0 else 's'
    
    ax1_twin.plot(x_positions, model_time_means, color=line_color, linestyle=line_style,
                  marker=marker_style, markersize=6, linewidth=2, alpha=0.8,
                  label=f'{model} (Time)')
    
    ax1_twin.errorbar(x_positions, model_time_means, yerr=model_time_stds, fmt='none',
                      ecolor=line_color, capsize=2, capthick=1, linewidth=1, alpha=0.6)

best_method_idx = 0
best_method = methods_sorted[0]
best_x = x_positions[best_method_idx]
best_y = max([method_stats[best_method]['accuracy'][model]['mean'] for model in models if model in method_stats[best_method]['accuracy']])

ax1.plot(best_x, best_y + 2, marker='*', markersize=15, color='gold', 
         markeredgecolor='black', markeredgewidth=2, zorder=10, label='Best performing')

ax1.set_ylim(10, 45)
ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
ax1_twin.set_ylabel('Time (s)', fontweight='bold', fontsize=12)
ax1.set_xlabel('Method', fontweight='bold', fontsize=12)

ax1.set_xticks(x_positions)
ax1.set_xticklabels(methods_sorted, fontsize=12)
ax1.tick_params(axis='both', labelsize=11)
ax1.grid(True, alpha=0.3, axis='y', linewidth=0.5)
apply_standard_plot_formatting(ax1, 'a', background_color=background_colors['white'])
ax1.set_title('Performance Comparison by Method and Model', fontsize=14, fontweight='bold', pad=20)

method_legend_elements = []
for method in methods_sorted:
    method_legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=method_colors[method], alpha=0.8, 
                                              edgecolor='black', linewidth=1.5, label=method))

model_legend_elements = []
for i, model in enumerate(models):
    alpha_value = 0.8 if i == 0 else 0.4
    model_legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=alpha_value,
                                             edgecolor='black', linewidth=1.5, label=model))

time_legend_elements = []
for i, model in enumerate(models):
    line_color = 'darkblue' if i == 0 else 'darkred'
    line_style = '-' if i == 0 else '--'
    marker_style = 'o' if i == 0 else 's'
    time_legend_elements.append(plt.Line2D([0], [0], color=line_color, linestyle=line_style,
                                         marker=marker_style, markersize=6, linewidth=2,
                                         label=f'{model} (Time)'))

if best_method:
    special_legend_elements = [plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                                        markeredgecolor='black', markeredgewidth=2, markersize=15,
                                        label='Best performing', linestyle='None')]

method_legend = ax1.legend(handles=method_legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.7), 
                          title='Methods', fontsize=9, ncol=1,
                          frameon=True, fancybox=True, shadow=True, framealpha=1.0, 
                          facecolor='white', edgecolor='black', title_fontsize=10)

model_legend = ax1.legend(handles=model_legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.4),
                         title='Models (Accuracy)', fontsize=9, ncol=1,
                         frameon=True, fancybox=True, shadow=True, framealpha=1.0,
                         facecolor='white', edgecolor='black', title_fontsize=10)

time_legend = ax1.legend(handles=time_legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.2),
                        title='Models (Time)', fontsize=9, ncol=1,
                        frameon=True, fancybox=True, shadow=True, framealpha=1.0,
                        facecolor='white', edgecolor='black', title_fontsize=10)

if best_method:
    special_legend = ax1.legend(handles=special_legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.05),
                               title='Special', fontsize=9, ncol=1,
                               frameon=True, fancybox=True, shadow=True, framealpha=1.0,
                               facecolor='white', edgecolor='black', title_fontsize=10)

ax1.add_artist(method_legend)
ax1.add_artist(model_legend)
ax1.add_artist(time_legend)

ax2 = fig.add_subplot(gs[1, 1:])

dataset_performance = df.groupby(['dataset', 'method']).agg({
    'accuracy': 'mean',
    'avg_time': 'mean',
    'avg_cost': 'mean'
}).reset_index()

datasets_sorted = sorted(datasets)

x_positions = {method: i for i, method in enumerate(methods)}
y_positions = {dataset: i for i, dataset in enumerate(datasets_sorted)}

for _, row in dataset_performance.iterrows():
    if row['method'] in methods:
        x = x_positions[row['method']]
        y = y_positions[row['dataset']]
        
        bubble_size = (row['accuracy'] / 50) * 1000
        bubble_alpha = min(1.0, max(0.5, row['accuracy'] / 50))
        
        time_normalized = min(1.0, row['avg_time'] / 60)
        edge_width = 1 + time_normalized * 3
        
        cost_normalized = min(1.0, (row['avg_cost'] * 100) / 50)
        edge_alpha = 0.3 + cost_normalized * 0.7
        
        bubble = ax2.scatter(x, y, 
                           s=bubble_size,
                           c=method_colors[row['method']], 
                           alpha=bubble_alpha,
                           edgecolors='black',
                           linewidths=edge_width,
                           zorder=5)
        
        if row['accuracy'] > 0:
            ax2.annotate(f"{row['accuracy']:.1f}%", 
                        (x, y), 
                        xytext=(0, 0), 
                        textcoords='offset points',
                        ha='center', 
                        va='center',
                        fontsize=8,
                        fontweight='bold',
                        color='black',
                        alpha=0.7,
                        zorder=10)

for i, method in enumerate(methods):
    ax2.axvline(x=i, color='gray', alpha=0.2, linewidth=0.5, zorder=1)

for i, dataset in enumerate(datasets_sorted):
    ax2.axhline(y=i, color='gray', alpha=0.2, linewidth=0.5, zorder=1)

ax2.set_xlim(-0.5, len(methods) - 0.5)
ax2.set_ylim(-0.5, len(datasets_sorted) - 0.5)

ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=12)
ax2.set_yticks(range(len(datasets_sorted)))
ax2.set_yticklabels(datasets_sorted, fontsize=12)

ax2.set_xlabel('Method', fontweight='bold', fontsize=12)
ax2.set_ylabel('Dataset', fontweight='bold', fontsize=12)
ax2.tick_params(axis='both', labelsize=11)

legend_elements = []
for size, label in [(200, '10%'), (500, '25%'), (800, '40%')]:
    legend_elements.append(plt.scatter([], [], s=size, c='gray', alpha=0.6, 
                                     edgecolors='black', linewidths=1, label=f'{label} Accuracy'))

legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=1, label='Low Time'))
legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=4, label='High Time'))

ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
          title='Bubble Size: Accuracy\nEdge Width: Time', fontsize=9, title_fontsize=10,
          frameon=True, fancybox=True, shadow=True, framealpha=1.0,
          facecolor='white', edgecolor='black')

apply_standard_plot_formatting(ax2, 'c', background_color=background_colors['white'])
ax2.set_title('Method Performance Matrix Across Datasets', fontsize=14, fontweight='bold', pad=20)

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
                   s=400, alpha=0.8, 
                   edgecolors='black', linewidth=2,
                   label=row['method'], marker='*')
    else:
        ax4.scatter(row['avg_cost'], row['accuracy'], 
                   c=method_colors[row['method']], 
                   s=200, alpha=0.8, 
                   edgecolors='black', linewidth=2,
                   label=row['method'])

pareto_indices = []
sorted_indices = np.argsort(avg_metrics['avg_cost'])
max_accuracy_so_far = -1
for i in sorted_indices:
    if avg_metrics.iloc[i]['accuracy'] > max_accuracy_so_far:
        pareto_indices.append(i)
        max_accuracy_so_far = avg_metrics.iloc[i]['accuracy']

if len(pareto_indices) > 1:
    pareto_data = avg_metrics.iloc[pareto_indices].sort_values('avg_cost')
    ax4.plot(pareto_data['avg_cost'], pareto_data['accuracy'], 'k--', alpha=0.7, linewidth=2, label='Pareto Frontier')

for _, row in avg_metrics.iterrows():
    ax4.annotate(row['method'], (row['avg_cost'], row['accuracy']), 
                xytext=(8, 8), textcoords='offset points', 
                fontsize=10, ha='left', va='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

ax4.set_xlabel('Average Cost (¢)', fontweight='bold', fontsize=12)
ax4.set_ylabel('Average Accuracy (%)', fontweight='bold', fontsize=12)
ax4.set_ylim(16, 32)
ax4.tick_params(axis='both', labelsize=11)
ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
apply_standard_plot_formatting(ax4, 'b', background_color=background_colors['white'])
ax4.set_title('Cost-Accuracy Trade-off', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('main_comparison.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()