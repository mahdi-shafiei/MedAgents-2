import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from plot_style import set_plot_style, get_color_scheme, get_marker_styles, get_line_styles, apply_standard_plot_formatting, get_standard_figure_size, get_standard_gridspec_params, get_background_colors, get_colors

set_plot_style()

df = pd.read_csv('agent_configuration.csv')
role_play_df = pd.read_csv('role_play.csv')
orchestration_df = pd.read_csv('orchestration_style.csv')

def parse_exp_name(exp_name):
    parts = exp_name.split('_')
    n_agents = int(parts[0])
    n_rounds = int(parts[2])
    has_search = 'with_search' in exp_name
    return n_agents, n_rounds, has_search

df[['n_agents', 'n_rounds', 'has_search']] = df['exp_name'].apply(
    lambda x: pd.Series(parse_exp_name(x))
)

summary_df = df.groupby(['n_agents', 'n_rounds', 'has_search']).agg({
    'accuracy': ['mean', 'std'],
    'avg_time': ['mean', 'std'],
    'avg_cost': 'mean'
}).round(2)

summary_df.columns = ['accuracy_mean', 'accuracy_std', 'time_mean', 'time_std', 'cost_mean']
summary_df = summary_df.reset_index()

role_play_summary = role_play_df.groupby('exp_name').agg({
    'accuracy': ['mean', 'std'],
    'avg_time': ['mean', 'std'],
    'avg_cost': 'mean'
}).round(2)

role_play_summary.columns = ['accuracy_mean', 'accuracy_std', 'time_mean', 'time_std', 'cost_mean']
role_play_summary = role_play_summary.reset_index()

orchestration_summary = orchestration_df.groupby('exp_name').agg({
    'accuracy': ['mean', 'std'],
    'avg_time': ['mean', 'std'],
    'avg_cost': 'mean'
}).round(2)

orchestration_summary.columns = ['accuracy_mean', 'accuracy_std', 'time_mean', 'time_std', 'cost_mean']
orchestration_summary = orchestration_summary.reset_index()
orchestration_summary = orchestration_summary.sort_values('exp_name', key=lambda x: x.map({
    'group_chat_with_orchestrator': 0,
    'group_chat_voting_only': 1,
    'independent': 2,
    'one_on_one_sync': 3
}))


agent_mapping = {1: 0, 2: 1, 3: 2, 5: 3}
summary_df['n_agents_mapped'] = summary_df['n_agents'].map(agent_mapping)

fig = plt.figure(figsize=(18, 16))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1.2, 1.2, 1], 
                      hspace=0.4, wspace=0.4)

colors = get_color_scheme('figure_2')
marker_styles = get_marker_styles()
line_styles = get_line_styles()
background_colors = get_background_colors()
manchester_colors = get_colors('manchester_united')
ax1 = fig.add_subplot(gs[0, 0])

agent_data_search = summary_df[(summary_df['n_rounds'] == 1) & (summary_df['has_search'] == True)]
if not agent_data_search.empty:
    line = ax1.plot(agent_data_search['n_agents_mapped'], agent_data_search['accuracy_mean'], 
                   marker='o', linewidth=3, markersize=10,
                   color=colors['metrics']['accuracy'], alpha=0.8,
                   markeredgecolor='black', markeredgewidth=2, label='With search')
    
    ax1.fill_between(agent_data_search['n_agents_mapped'], 
                    agent_data_search['accuracy_mean'] - agent_data_search['accuracy_std'],
                    agent_data_search['accuracy_mean'] + agent_data_search['accuracy_std'],
                    color=colors['metrics']['accuracy'], alpha=0.2)
    
    for i, row in agent_data_search.iterrows():
        ax1.annotate(f'{row["accuracy_mean"]:.1f}%', 
                    (row['n_agents_mapped'], row['accuracy_mean']),
                    textcoords="offset points", xytext=(0,15), ha='center',
                    fontsize=9, fontweight='bold', color='black')

agent_data_no_search = summary_df[(summary_df['n_rounds'] == 1) & (summary_df['has_search'] == False)]
if not agent_data_no_search.empty:
    line = ax1.plot(agent_data_no_search['n_agents_mapped'], agent_data_no_search['accuracy_mean'], 
                   marker='s', linewidth=3, markersize=10,
                   color=colors['metrics']['time'], alpha=0.8,
                   markeredgecolor='black', markeredgewidth=2, label='No search')
    
    ax1.fill_between(agent_data_no_search['n_agents_mapped'], 
                    agent_data_no_search['accuracy_mean'] - agent_data_no_search['accuracy_std'],
                    agent_data_no_search['accuracy_mean'] + agent_data_no_search['accuracy_std'],
                    color=colors['metrics']['time'], alpha=0.2)
    
    for i, row in agent_data_no_search.iterrows():
        ax1.annotate(f'{row["accuracy_mean"]:.1f}%', 
                    (row['n_agents_mapped'], row['accuracy_mean']),
                    textcoords="offset points", xytext=(0,-20), ha='center',
                    fontsize=9, fontweight='bold', color='black')

best_accuracy_idx = summary_df[(summary_df['n_rounds'] == 1)]['accuracy_mean'].idxmax()
best_point = summary_df.loc[best_accuracy_idx]
best_x = agent_mapping[best_point['n_agents']]
best_y = best_point['accuracy_mean']
ax1.plot(best_x, best_y, marker='*', markersize=15, color='gold', 
         markeredgecolor='black', markeredgewidth=2, label='Best performing')

ax1.axhline(y=best_y, color='gold', linestyle='--', alpha=0.7, linewidth=2)

ax1.set_xlabel('Number of agents', fontweight='bold', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
ax1.tick_params(axis='both', labelsize=11)
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.legend(fontsize=9, loc='upper left', frameon=True, fancybox=True, shadow=True, 
           framealpha=1.0, facecolor='white', edgecolor='black')
apply_standard_plot_formatting(ax1, 'a', background_color=background_colors['white'])
ax1.set_xlim(-0.5, 3.5)
ax1.set_ylim(20, 45)
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels([1, 2, 3, 5])

ax2 = fig.add_subplot(gs[0, 1])

rounds_data_search = summary_df[(summary_df['n_agents'] == 3) & (summary_df['has_search'] == True)]
rounds_data_no_search = summary_df[(summary_df['n_agents'] == 3) & (summary_df['has_search'] == False)]

rounds = sorted(set(list(rounds_data_search['n_rounds']) + list(rounds_data_no_search['n_rounds'])))
x_pos = np.arange(len(rounds))
width = 0.35

search_accuracies = []
search_stds = []
no_search_accuracies = []
no_search_stds = []
search_times = []
search_time_stds = []
no_search_times = []
no_search_time_stds = []

for round_num in rounds:
    search_data = rounds_data_search[rounds_data_search['n_rounds'] == round_num]
    if not search_data.empty:
        search_accuracies.append(search_data['accuracy_mean'].iloc[0])
        search_stds.append(search_data['accuracy_std'].iloc[0])
        search_times.append(search_data['time_mean'].iloc[0])
        search_time_stds.append(search_data['time_std'].iloc[0])
    else:
        search_accuracies.append(0)
        search_stds.append(0)
        search_times.append(0)
        search_time_stds.append(0)
    
    no_search_data = rounds_data_no_search[rounds_data_no_search['n_rounds'] == round_num]
    if not no_search_data.empty:
        no_search_accuracies.append(no_search_data['accuracy_mean'].iloc[0])
        no_search_stds.append(no_search_data['accuracy_std'].iloc[0])
        no_search_times.append(no_search_data['time_mean'].iloc[0])
        no_search_time_stds.append(no_search_data['time_std'].iloc[0])
    else:
        no_search_accuracies.append(0)
        no_search_stds.append(0)
        no_search_times.append(0)
        no_search_time_stds.append(0)

bars1 = ax2.bar(x_pos - width/2, no_search_accuracies, width, 
                color=colors['metrics']['time'], alpha=0.8, 
                edgecolor='black', linewidth=1.5, label='No search')

bars2 = ax2.bar(x_pos + width/2, search_accuracies, width,
                color=colors['metrics']['accuracy'], alpha=0.8, 
                edgecolor='black', linewidth=1.5, label='With search')

ax2.errorbar(x_pos - width/2, no_search_accuracies, yerr=no_search_stds, 
             fmt='none', color='black', alpha=0.5, capsize=3, capthick=2)
ax2.errorbar(x_pos + width/2, search_accuracies, yerr=search_stds, 
             fmt='none', color='black', alpha=0.5, capsize=3, capthick=2)

for i, (bar, acc) in enumerate(zip(bars1, no_search_accuracies)):
    if acc > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

for i, (bar, acc) in enumerate(zip(bars2, search_accuracies)):
    if acc > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xlabel('Number of rounds', fontweight='bold', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
ax2.tick_params(axis='both', labelsize=11)
ax2.grid(True, alpha=0.3, axis='y', linewidth=0.5)
ax2.legend(fontsize=9, loc='upper left', frameon=True, fancybox=True, shadow=True, 
           framealpha=1.0, facecolor='white', edgecolor='black')
apply_standard_plot_formatting(ax2, 'b', background_color=background_colors['white'])
ax2.set_ylim(20, 45)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(rounds)

ax2_time = ax2.twinx()

time_line1 = ax2_time.plot(x_pos, [t for t in no_search_times if t > 0], 
                          marker='s', linewidth=2, markersize=8,
                          color=colors['metrics']['time'], alpha=0.6,
                          markeredgecolor='black', markeredgewidth=1, 
                          linestyle='--', label='No search (time)')

time_line2 = ax2_time.plot(x_pos, [t for t in search_times if t > 0], 
                          marker='o', linewidth=2, markersize=8,
                          color=colors['metrics']['accuracy'], alpha=0.6,
                          markeredgecolor='black', markeredgewidth=1, 
                          linestyle='--', label='With search (time)')

ax2_time.errorbar(x_pos, [t for t in no_search_times if t > 0], 
                 yerr=[std for std in no_search_time_stds if std > 0], 
                 fmt='none', color=colors['metrics']['time'], alpha=0.6, 
                 capsize=3, capthick=1)
ax2_time.errorbar(x_pos, [t for t in search_times if t > 0], 
                 yerr=[std for std in search_time_stds if std > 0], 
                 fmt='none', color=colors['metrics']['accuracy'], alpha=0.6, 
                 capsize=3, capthick=1)

ax2_time.set_ylabel('Average Time (s)', fontweight='bold', fontsize=12, color='gray')
ax2_time.tick_params(axis='y', labelcolor='gray', labelsize=10)
ax2_time.legend(fontsize=9, loc='upper right', frameon=True, fancybox=True, shadow=True, 
                framealpha=1.0, facecolor='white', edgecolor='black')

ax3 = fig.add_subplot(gs[0, 2])

role_play_conditions = ['enable_role_play', 'disable_role_play']
role_play_labels = ['Yes', 'No']
x_pos = np.arange(len(role_play_conditions))

role_play_accuracies = []
role_play_stds = []
role_play_times = []
role_play_time_stds = []

for condition in role_play_conditions:
    condition_data = role_play_summary[role_play_summary['exp_name'] == condition]
    if not condition_data.empty:
        role_play_accuracies.append(condition_data['accuracy_mean'].iloc[0])
        role_play_stds.append(condition_data['accuracy_std'].iloc[0])
        role_play_times.append(condition_data['time_mean'].iloc[0])
        role_play_time_stds.append(condition_data['time_std'].iloc[0])
    else:
        role_play_accuracies.append(0)
        role_play_stds.append(0)
        role_play_times.append(0)
        role_play_time_stds.append(0)

bars = ax3.bar(x_pos, role_play_accuracies, 
               color=[manchester_colors['red_dark'], manchester_colors['gray_medium']], 
               alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.errorbar(x_pos, role_play_accuracies, yerr=role_play_stds, 
             fmt='none', color='black', alpha=0.5, capsize=3, capthick=1)

for i, (bar, acc) in enumerate(zip(bars, role_play_accuracies)):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_xlabel('Role Play', fontweight='bold', fontsize=12)
ax3.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
ax3.tick_params(axis='both', labelsize=11)
ax3.grid(True, alpha=0.2, axis='y', linewidth=0.8, linestyle='-')
apply_standard_plot_formatting(ax3, 'c', background_color=background_colors['white'])
ax3.set_ylim(20, 45)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(role_play_labels)

ax3_time = ax3.twinx()

time_line = ax3_time.plot(x_pos, role_play_times, 
                         marker='o', linewidth=2.5, markersize=8,
                         color=colors['metrics']['time'], alpha=0.7,
                         markeredgecolor='black', markeredgewidth=1.5, 
                         linestyle='--', label='Processing Time')

ax3_time.errorbar(x_pos, role_play_times, yerr=role_play_time_stds, 
                 fmt='none', color=colors['metrics']['time'], alpha=0.7, 
                 capsize=3, capthick=2)

ax3_time.set_ylabel('Processing Time (s)', fontweight='bold', fontsize=12, color='gray')
ax3_time.tick_params(axis='y', labelcolor='gray', labelsize=10)
ax3_time.legend(fontsize=9, loc='upper right', frameon=True, fancybox=True, shadow=True, 
                framealpha=1.0, facecolor='white', edgecolor='black')

time_reduction = ((role_play_times[0] - role_play_times[1]) / role_play_times[0] * 100) if len(role_play_times) > 1 and role_play_times[0] > 0 else 0
if time_reduction > 0:
    ax3.text(0.98, 0.02, f'Time Reduction: {time_reduction:.1f}%', 
             transform=ax3.transAxes, fontsize=9, fontweight='bold',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['metrics']['time'], 
                      alpha=0.8, edgecolor='white'))

ax4 = fig.add_subplot(gs[1, 0])

baseline_accuracy = summary_df[(summary_df['n_agents'] == 1) & 
                              (summary_df['n_rounds'] == 1) & 
                              (summary_df['has_search'] == False)]['accuracy_mean'].iloc[0]
baseline_cost = summary_df[(summary_df['n_agents'] == 1) & 
                          (summary_df['n_rounds'] == 1) & 
                          (summary_df['has_search'] == False)]['cost_mean'].iloc[0]

improvements = []
improvement_labels = []
improvement_colors = []
improvement_descriptions = []

add_agents_data = summary_df[(summary_df['n_agents'] == 3) & 
                            (summary_df['n_rounds'] == 1) & 
                            (summary_df['has_search'] == False)]
if not add_agents_data.empty:
    agents_accuracy = add_agents_data['accuracy_mean'].iloc[0]
    agents_cost = add_agents_data['cost_mean'].iloc[0]
    agents_improvement = ((agents_accuracy - baseline_accuracy) / (agents_cost - baseline_cost)) if agents_cost != baseline_cost else 0
    improvements.append(agents_improvement)
    improvement_labels.append('Multi-Agent')
    improvement_colors.append(manchester_colors['red_dark'])
    improvement_descriptions.append('Multi-Agent')

add_rounds_data = summary_df[(summary_df['n_agents'] == 1) & 
                            (summary_df['n_rounds'] == 3) & 
                            (summary_df['has_search'] == True)]
if not add_rounds_data.empty:
    rounds_accuracy = add_rounds_data['accuracy_mean'].iloc[0]
    rounds_cost = add_rounds_data['cost_mean'].iloc[0]
    rounds_improvement = ((rounds_accuracy - baseline_accuracy) / (rounds_cost - baseline_cost)) if rounds_cost != baseline_cost else 0
    improvements.append(rounds_improvement)
    improvement_labels.append('Iterative')
    improvement_colors.append(manchester_colors['gold_dark'])
    improvement_descriptions.append('Iterative')

add_search_data = summary_df[(summary_df['n_agents'] == 1) & 
                            (summary_df['n_rounds'] == 1) & 
                            (summary_df['has_search'] == True)]
if not add_search_data.empty:
    search_accuracy = add_search_data['accuracy_mean'].iloc[0]
    search_cost = add_search_data['cost_mean'].iloc[0]
    search_improvement = ((search_accuracy - baseline_accuracy) / (search_cost - baseline_cost)) if search_cost != baseline_cost else 0
    improvements.append(search_improvement)
    improvement_labels.append('Evidence')
    improvement_colors.append(manchester_colors['gold_medium'])
    improvement_descriptions.append('Evidence')

disable_role_play_data = role_play_summary[role_play_summary['exp_name'] == 'disable_role_play']
enable_role_play_data = role_play_summary[role_play_summary['exp_name'] == 'enable_role_play']

if not disable_role_play_data.empty and not enable_role_play_data.empty:
    disable_accuracy = disable_role_play_data['accuracy_mean'].iloc[0]
    disable_cost = disable_role_play_data['cost_mean'].iloc[0]
    enable_accuracy = enable_role_play_data['accuracy_mean'].iloc[0]
    enable_cost = enable_role_play_data['cost_mean'].iloc[0]

    role_play_improvement = ((enable_accuracy - disable_accuracy) / (enable_cost - disable_cost)) if enable_cost != disable_cost else 0
    improvements.append(role_play_improvement)
    improvement_labels.append('Role Play')
    improvement_colors.append(manchester_colors['black'])
    improvement_descriptions.append('Role Play')

orchestration_baseline_data = orchestration_summary[orchestration_summary['exp_name'] == 'independent']
orchestration_best_data = orchestration_summary[orchestration_summary['exp_name'] == 'group_chat_with_orchestrator']

if not orchestration_baseline_data.empty and not orchestration_best_data.empty:
    orchestration_baseline_accuracy = orchestration_baseline_data['accuracy_mean'].iloc[0]
    orchestration_baseline_cost = orchestration_baseline_data['cost_mean'].iloc[0]
    orchestration_best_accuracy = orchestration_best_data['accuracy_mean'].iloc[0]
    orchestration_best_cost = orchestration_best_data['cost_mean'].iloc[0]

    orchestration_improvement = ((orchestration_best_accuracy - orchestration_baseline_accuracy) / 
                               (orchestration_best_cost - orchestration_baseline_cost)) if orchestration_best_cost != orchestration_baseline_cost else 0
    improvements.append(orchestration_improvement)
    improvement_labels.append('Discussion')
    improvement_colors.append(manchester_colors['gray_dark'])
    improvement_descriptions.append('Discussion')

bars = ax4.bar(range(len(improvements)), improvements, color=improvement_colors, 
               alpha=0.8, edgecolor='black', linewidth=2)

for i, (bar, improvement) in enumerate(zip(bars, improvements)):
    height = bar.get_height()
    y_pos = height + (abs(height) * 0.05 if height != 0 else 0.1)
    ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
             f'{improvement:.1f}', ha='center', 
             va='bottom' if height >= 0 else 'top', 
             fontsize=11, fontweight='bold', color='black')

ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

best_idx = improvements.index(max(improvements)) if improvements else 0
if improvements:
    ax4.plot(best_idx, improvements[best_idx], marker='*', markersize=15, 
             color='gold', markeredgecolor='black', markeredgewidth=2, 
             zorder=10, label='Most effective')

ax4.set_xlabel('Enhancement Strategy', fontweight='bold', fontsize=12)
ax4.set_ylabel('Cost-Effectiveness\n(Δ Accuracy % / Δ Cost ¢)', fontweight='bold', fontsize=12)
ax4.tick_params(axis='both', labelsize=11)
ax4.grid(True, alpha=0.2, axis='y', linewidth=0.8, linestyle='-')
apply_standard_plot_formatting(ax4, 'd', background_color=background_colors['white'])

if improvements:
    y_min = min(improvements)
    y_max = max(improvements)
    y_range = y_max - y_min
    padding = y_range * 0.2
    ax4.set_ylim(y_min - padding, y_max + padding)
else:
    ax4.set_ylim(-1, 1)

ax4.set_xlim(-0.6, len(improvements) - 0.4)
ax4.set_xticks(range(len(improvements)))
ax4.set_xticklabels(improvement_labels, fontsize=10, ha='center')
ax4.legend(loc='upper left', fontsize=9, 
           frameon=True, fancybox=True, shadow=True, framealpha=1.0, 
           facecolor='white', edgecolor='black')

ax5 = fig.add_subplot(gs[1, 1])

orchestration_conditions = orchestration_summary['exp_name'].unique()
orchestration_labels = ['Group Chat\n+ Orchestrator', 'Group Chat', 'Independent', 'One-on-One\n+ Orchestrator']
x_pos = np.arange(len(orchestration_conditions))

orchestration_accuracies = []
orchestration_stds = []
orchestration_times = []
orchestration_time_stds = []

for condition in orchestration_conditions:
    condition_data = orchestration_summary[orchestration_summary['exp_name'] == condition]
    if not condition_data.empty:
        orchestration_accuracies.append(condition_data['accuracy_mean'].iloc[0])
        orchestration_stds.append(condition_data['accuracy_std'].iloc[0])
        orchestration_times.append(condition_data['time_mean'].iloc[0])
        orchestration_time_stds.append(condition_data['time_std'].iloc[0])
    else:
        orchestration_accuracies.append(0)
        orchestration_stds.append(0)
        orchestration_times.append(0)
        orchestration_time_stds.append(0)

orchestration_colors = [
    manchester_colors['red_dark'],
    manchester_colors['gold_dark'],
    manchester_colors['gray_medium'],
    manchester_colors['gold_dark']
]

bars = ax5.bar(x_pos, orchestration_accuracies, 
               color=orchestration_colors, 
               alpha=0.8, edgecolor='black', linewidth=1.5)
ax5.errorbar(x_pos, orchestration_accuracies, yerr=orchestration_stds, 
             fmt='none', color='black', alpha=0.5, capsize=3, capthick=2)

for i, (bar, acc) in enumerate(zip(bars, orchestration_accuracies)):
    ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax5.set_xlabel('Discussion Strategy', fontweight='bold', fontsize=12)
ax5.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
ax5.tick_params(axis='both', labelsize=11)
ax5.grid(True, alpha=0.2, axis='y', linewidth=0.8, linestyle='-')
apply_standard_plot_formatting(ax5, 'e', background_color=background_colors['white'])
ax5.set_ylim(20, 45)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(orchestration_labels, fontsize=10)

ax5_time = ax5.twinx()

time_line = ax5_time.plot(x_pos, orchestration_times, 
                         marker='o', linewidth=2.5, markersize=8,
                         color=colors['metrics']['time'], alpha=0.7,
                         markeredgecolor='black', markeredgewidth=1.5, 
                         linestyle='--', label='Processing Time')

ax5_time.errorbar(x_pos, orchestration_times, yerr=orchestration_time_stds, 
                 fmt='none', color=colors['metrics']['time'], alpha=0.7, 
                 capsize=3, capthick=2)

ax5_time.set_ylabel('Processing Time (s)', fontweight='bold', fontsize=12, color='gray')
ax5_time.tick_params(axis='y', labelcolor='gray', labelsize=10)
ax5_time.legend(fontsize=9, loc='upper right', frameon=True, fancybox=True, shadow=True, 
                framealpha=1.0, facecolor='white', edgecolor='black')

best_orchestration_idx = orchestration_accuracies.index(max(orchestration_accuracies)) if orchestration_accuracies else 0
if orchestration_accuracies:
    ax5.plot(best_orchestration_idx, orchestration_accuracies[best_orchestration_idx], 
             marker='*', markersize=15, color='gold', 
             markeredgecolor='black', markeredgewidth=2, 
             zorder=10, label='Best performing')
    ax5.legend(fontsize=9, loc='upper left', frameon=True, fancybox=True, shadow=True, 
               framealpha=1.0, facecolor='white', edgecolor='black')

plt.tight_layout(pad=3.0)
plt.savefig('agent_configuration_analysis.pdf', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')