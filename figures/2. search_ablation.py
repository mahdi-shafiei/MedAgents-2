import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from plot_style import set_plot_style, get_color_scheme, get_marker_styles, get_line_styles, apply_standard_plot_formatting, get_standard_figure_size, get_standard_gridspec_params, get_background_colors, get_colors

set_plot_style()

df = pd.read_csv('search_ablation.csv')

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1.5], 
                      hspace=0.4, wspace=0.5)

colors = get_color_scheme('figure_2')
marker_styles = get_marker_styles()
line_styles = get_line_styles()
background_colors = get_background_colors()
manchester_colors = get_colors('manchester_united')

ax1 = fig.add_subplot(gs[0, 0])

modality_df = df[df['ablation'] == 'search_modality']
modality_names = ['Search with\nWeb & Vector', 'Search with\nVector Only', 'Search with\nWeb Only', 'No Search']
accuracies = modality_df['accuracy'].values
times = modality_df['avg_time'].values
costs = modality_df['avg_cost'].values

bars1 = ax1.bar(np.arange(len(modality_names)) - 0.2, accuracies, 0.4, 
                color=colors['metrics']['accuracy'], alpha=0.8, 
                label='Accuracy (%)', edgecolor='black', linewidth=1.5)

ax1_twin = ax1.twinx()
bars2 = ax1_twin.bar(np.arange(len(modality_names)) + 0.2, times, 0.4,
                     color=colors['metrics']['time'], alpha=0.8,
                     label='Time (s)', edgecolor='black', linewidth=1.5)

best_accuracy = accuracies[0]
for i, (bar, acc, cost) in enumerate(zip(bars1, accuracies, costs)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{acc:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.text(bar.get_x() + bar.get_width()/2., height/2,
             f'{cost:.1f}¢', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    if i > 0:
        reduction = ((best_accuracy - acc) / best_accuracy) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height + 3,
                 f'↓{reduction:.0f}%', ha='center', va='bottom', fontsize=10, 
                 fontweight='bold', color='grey', 
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgrey', alpha=0.7))

for i, (bar, time) in enumerate(zip(bars2, times)):
    height = bar.get_height()
    ax1_twin.text(bar.get_x() + bar.get_width()/2., height + 5,
                  f'{time:.0f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xlabel('Search Modality', fontweight='bold', fontsize=11)
ax1.set_ylabel('Accuracy (%)', fontweight='bold', color=colors['metrics']['accuracy'], fontsize=11)
ax1_twin.set_ylabel('Time (s)', fontweight='bold', color=colors['metrics']['time'], fontsize=11)
ax1.set_xticks(np.arange(len(modality_names)))
ax1.set_xticklabels(modality_names, fontsize=10)
ax1.tick_params(axis='both', labelsize=10)
ax1.set_ylim(0, 40)
ax1_twin.set_ylim(0, 400)
apply_standard_plot_formatting(ax1, 'a', background_color=background_colors['white'])
ax1.set_title('Search Modality Comparison', fontsize=12, fontweight='bold', pad=35)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)

ax2 = fig.add_subplot(gs[0, 1])

features_df = df[df['ablation'] == 'search_features']
feature_names = ['Full Search', 'w/o Doc Review', 'w/o Query Rewrite', 'w/o Doc Review\nw/o Query Rewrite']
feature_accuracies = features_df['accuracy'].values
feature_times = features_df['avg_time'].values
feature_costs = features_df['avg_cost'].values

bars1 = ax2.bar(np.arange(len(feature_names)) - 0.2, feature_accuracies, 0.4,
                color=colors['metrics']['accuracy'], alpha=0.8, 
                label='Accuracy (%)', edgecolor='black', linewidth=1.5)

ax2_twin = ax2.twinx()
bars2 = ax2_twin.bar(np.arange(len(feature_names)) + 0.2, feature_times, 0.4,
                     color=colors['metrics']['time'], alpha=0.8,
                     label='Time (s)', edgecolor='black', linewidth=1.5)

best_feature_accuracy = feature_accuracies[0]
for i, (bar, acc, cost) in enumerate(zip(bars1, feature_accuracies, feature_costs)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{acc:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.text(bar.get_x() + bar.get_width()/2., height/2,
             f'{cost:.1f}¢', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    if i > 0:
        reduction = ((best_feature_accuracy - acc) / best_feature_accuracy) * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height + 3,
                 f'↓{reduction:.0f}%', ha='center', va='bottom', fontsize=10, 
                 fontweight='bold', color='grey', 
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgrey', alpha=0.7))

for i, (bar, time) in enumerate(zip(bars2, feature_times)):
    height = bar.get_height()
    ax2_twin.text(bar.get_x() + bar.get_width()/2., height + 10,
                  f'{time:.0f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xlabel('Search Features', fontweight='bold', fontsize=11)
ax2.set_ylabel('Accuracy (%)', fontweight='bold', color=colors['metrics']['accuracy'], fontsize=11)
ax2_twin.set_ylabel('Time (s)', fontweight='bold', color=colors['metrics']['time'], fontsize=11)
ax2.set_xticks(np.arange(len(feature_names)))
ax2.set_xticklabels(feature_names, fontsize=10)
ax2.tick_params(axis='both', labelsize=10)
ax2.set_ylim(0, 40)
ax2_twin.set_ylim(0, 600)
apply_standard_plot_formatting(ax2, 'b', background_color=background_colors['white'])
ax2.set_title('Search Features Ablation', fontsize=12, fontweight='bold', pad=35)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)

ax4 = fig.add_subplot(gs[1, 0])

history_df = df[df['ablation'] == 'search_history']
history_names = ['Shared History', 'Individual History']
history_accuracies = history_df['accuracy'].values[::-1]
history_times = history_df['avg_time'].values[::-1]
history_costs = history_df['avg_cost'].values[::-1]

x_pos = np.arange(len(history_names))
bars1 = ax4.bar(x_pos - 0.2, history_accuracies, 0.4, 
                color=colors['metrics']['accuracy'], alpha=0.8, 
                label='Accuracy (%)', edgecolor='black', linewidth=1.5)

ax4_twin = ax4.twinx()
bars2 = ax4_twin.bar(x_pos + 0.2, history_times, 0.4,
                     color=colors['metrics']['time'], alpha=0.8,
                     label='Time (s)', edgecolor='black', linewidth=1.5)

best_history_accuracy = history_accuracies[0]
for i, (bar, acc, cost) in enumerate(zip(bars1, history_accuracies, history_costs)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{acc:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax4.text(bar.get_x() + bar.get_width()/2., height/2,
             f'{cost:.1f}¢', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    if i > 0:
        reduction = ((best_history_accuracy - acc) / best_history_accuracy) * 100
        ax4.text(bar.get_x() + bar.get_width()/2., height + 3,
                 f'↓{reduction:.0f}%', ha='center', va='bottom', fontsize=10, 
                 fontweight='bold', color='grey', 
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgrey', alpha=0.7))

for i, (bar, time) in enumerate(zip(bars2, history_times)):
    height = bar.get_height()
    ax4_twin.text(bar.get_x() + bar.get_width()/2., height + 30,
                  f'{time:.0f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax4.set_xlabel('History Strategy', fontweight='bold', fontsize=11)
ax4.set_ylabel('Accuracy (%)', fontweight='bold', color=colors['metrics']['accuracy'], fontsize=11)
ax4_twin.set_ylabel('Time (s)', fontweight='bold', color=colors['metrics']['time'], fontsize=11)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(history_names, fontsize=10)
ax4.tick_params(axis='both', labelsize=10)
ax4.set_ylim(0, 40)
ax4_twin.set_ylim(0, 1800)
apply_standard_plot_formatting(ax4, 'c', background_color=background_colors['white'])
ax4.set_title('Search History Strategy', fontsize=12, fontweight='bold', pad=35)

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)

ax6 = fig.add_subplot(gs[1, 1])

modality_data = modality_df[['exp_name', 'accuracy', 'avg_cost']].copy()
modality_data['ablation_type'] = 'Modality'
features_data = features_df[['exp_name', 'accuracy', 'avg_cost']].copy()
features_data['ablation_type'] = 'Features'
history_data = history_df[['exp_name', 'accuracy', 'avg_cost']].copy()
history_data['ablation_type'] = 'History'

all_data = pd.concat([modality_data, features_data, history_data], ignore_index=True)
all_data['cost_cents'] = all_data['avg_cost']

def get_search_modality(exp_name):
    if exp_name == 'both':
        return 'both'
    elif exp_name in ['vector_only', 'baseline', 'no_document_review', 'no_query_rewrite', 'no_rewrite_no_review']:
        return 'vector'
    elif exp_name == 'web_only':
        return 'web'
    elif exp_name in ['shared', 'individual']:
        return 'both'
    else:
        return 'both'

def get_feature_level(exp_name):
    if exp_name in ['both', 'baseline', 'shared', 'individual']:
        return 'full'
    elif exp_name == 'no_document_review':
        return 'no_doc_review'
    elif exp_name == 'no_query_rewrite':
        return 'no_query_rewrite'
    elif exp_name == 'no_rewrite_no_review':
        return 'minimal'
    else:
        return 'full'

def get_history_type(exp_name):
    if exp_name in ['shared']:
        return 'shared'
    else:
        return 'individual'

all_data['search_modality'] = all_data['exp_name'].apply(get_search_modality)
all_data['feature_level'] = all_data['exp_name'].apply(get_feature_level)
all_data['history_type'] = all_data['exp_name'].apply(get_history_type)

modality_shapes = {
    'both': 'o',
    'vector': 's',
    'web': '^'
}

feature_sizes = {
    'full': 120,
    'no_doc_review': 100,
    'no_query_rewrite': 80,
    'minimal': 60
}

history_colors = {
    'individual': manchester_colors['red_dark'],
    'shared': manchester_colors['gold_dark']
}

for _, row in all_data.iterrows():
    ax6.scatter(row['cost_cents'], row['accuracy'], 
               c=history_colors[row['history_type']], 
               marker=modality_shapes[row['search_modality']],
               alpha=0.8, s=feature_sizes[row['feature_level']], 
               edgecolors='black', linewidth=1.5)

sorted_indices = np.argsort(all_data['cost_cents'])
sorted_costs = all_data['cost_cents'].iloc[sorted_indices]
sorted_accuracies = all_data['accuracy'].iloc[sorted_indices]
sorted_names = all_data['exp_name'].iloc[sorted_indices]

pareto_indices = []
max_accuracy_so_far = -1
for i, (cost, accuracy) in enumerate(zip(sorted_costs, sorted_accuracies)):
    if accuracy > max_accuracy_so_far:
        pareto_indices.append(sorted_indices.iloc[i])
        max_accuracy_so_far = accuracy

pareto_data = all_data.iloc[pareto_indices]
pareto_sort = np.argsort(pareto_data['cost_cents'])
pareto_sorted = pareto_data.iloc[pareto_sort]

ax6.plot(pareto_sorted['cost_cents'], pareto_sorted['accuracy'], 'k--', alpha=0.7, linewidth=2, label='Pareto Frontier')

label_mapping = {
    'both': 'Web+Vector',
    'vector_only': 'Vector Only',
    'web_only': 'Web Only',
    'baseline': 'Full Vector Search',
    'no_query_rewrite': 'No Query Rewrite',
    'no_rewrite_no_review': 'Minimal Search',
    'shared': 'Shared History'
}

for _, row in pareto_sorted.iterrows():
    if row['exp_name'] in label_mapping:
        label = label_mapping[row['exp_name']]
        ax6.annotate(label, (row['cost_cents'], row['accuracy']), 
                    xytext=(8, 8), textcoords='offset points', 
                fontsize=9, ha='left', va='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

for _, row in all_data.iterrows():
    if row.name not in pareto_indices:
        if row['exp_name'] in label_mapping:
            label = label_mapping[row['exp_name']]
            ax6.annotate(label, (row['cost_cents'], row['accuracy']), 
                        xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, ha='left', va='bottom', alpha=0.7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.6))

legend_elements = []
legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Web+Vector'))
legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, label='Vector Only'))
legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Web Only'))
legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=history_colors['individual'], markersize=8, label='Individual History'))
legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=history_colors['shared'], markersize=8, label='Shared History'))

ax6.set_xlabel('Cost (¢)', fontweight='bold', fontsize=11)
ax6.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=11)
ax6.tick_params(axis='both', labelsize=10)
ax6.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
apply_standard_plot_formatting(ax6, 'd', background_color=background_colors['white'])
ax6.set_title('Cost-Accuracy Pareto Analysis', fontsize=12, fontweight='bold', pad=35)

ax6.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fontsize=9, frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('search_ablation.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
