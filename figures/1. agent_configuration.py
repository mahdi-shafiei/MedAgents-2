import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from plot_style import set_plot_style, get_color_scheme, get_marker_styles, get_line_styles, apply_standard_plot_formatting, get_standard_figure_size, get_standard_gridspec_params, get_background_colors, get_dataset_colors

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

fig = plt.figure(figsize=(26, 24))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1.2, 1.0, 0.6], 
                      hspace=0.3, wspace=0.3)

colors = get_color_scheme('figure_1', theme='manchester_united_official')
marker_styles = get_marker_styles()
line_styles = get_line_styles()
background_colors = get_background_colors()
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
                    fontsize=12, fontweight='bold', color='black')

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
                    fontsize=12, fontweight='bold', color='black')

best_accuracy_idx = summary_df[(summary_df['n_rounds'] == 1)]['accuracy_mean'].idxmax()
best_point = summary_df.loc[best_accuracy_idx]
best_x = agent_mapping[best_point['n_agents']]
best_y = best_point['accuracy_mean']
ax1.plot(best_x, best_y, marker='*', markersize=15, color='gold', 
         markeredgecolor='black', markeredgewidth=2, label='Best performing')

ax1.axhline(y=best_y, color='gold', linestyle='--', alpha=0.7, linewidth=2)

ax1.set_xlabel('Number of agents', fontweight='bold', fontsize=16)
ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
ax1.tick_params(axis='both', labelsize=14)
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.legend(fontsize=12, loc='upper left', frameon=True, fancybox=True, shadow=True, 
           framealpha=1.0, facecolor='white', edgecolor='black')
apply_standard_plot_formatting(ax1, 'a', background_color=background_colors['white'], fontsize=20)
ax1.set_xlim(-0.5, 3.5)
ax1.set_ylim(20, 45)
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels([1, 2, 3, 5])

ax2 = fig.add_subplot(gs[0, 1])

rounds_data_search = summary_df[(summary_df['n_agents'] == 3) & (summary_df['has_search'] == True)]
rounds_data_no_search = summary_df[(summary_df['n_agents'] == 3) & (summary_df['has_search'] == False)]

if not rounds_data_search.empty:
    line = ax2.plot(rounds_data_search['n_rounds'], rounds_data_search['accuracy_mean'], 
                   marker='o', linewidth=3, markersize=10,
                   color=colors['metrics']['accuracy'], alpha=0.8,
                   markeredgecolor='black', markeredgewidth=2, label='With search')
    
    ax2.fill_between(rounds_data_search['n_rounds'], 
                    rounds_data_search['accuracy_mean'] - rounds_data_search['accuracy_std'],
                    rounds_data_search['accuracy_mean'] + rounds_data_search['accuracy_std'],
                    color=colors['metrics']['accuracy'], alpha=0.2)
    
    for i, row in rounds_data_search.iterrows():
        ax2.annotate(f'{row["accuracy_mean"]:.1f}%', 
                    (row['n_rounds'], row['accuracy_mean']),
                    textcoords="offset points", xytext=(0,15), ha='center',
                    fontsize=12, fontweight='bold', color='black')

if not rounds_data_no_search.empty:
    line = ax2.plot(rounds_data_no_search['n_rounds'], rounds_data_no_search['accuracy_mean'], 
                   marker='s', linewidth=3, markersize=10,
                   color=colors['metrics']['time'], alpha=0.8,
                   markeredgecolor='black', markeredgewidth=2, label='No search')
    
    ax2.fill_between(rounds_data_no_search['n_rounds'], 
                    rounds_data_no_search['accuracy_mean'] - rounds_data_no_search['accuracy_std'],
                    rounds_data_no_search['accuracy_mean'] + rounds_data_no_search['accuracy_std'],
                    color=colors['metrics']['time'], alpha=0.2)
    
    for i, row in rounds_data_no_search.iterrows():
        ax2.annotate(f'{row["accuracy_mean"]:.1f}%', 
                    (row['n_rounds'], row['accuracy_mean']),
                    textcoords="offset points", xytext=(0,-20), ha='center',
                    fontsize=12, fontweight='bold', color='black')

best_accuracy_idx = summary_df[(summary_df['n_agents'] == 3)]['accuracy_mean'].idxmax()
best_point = summary_df.loc[best_accuracy_idx]
best_x = best_point['n_rounds']
best_y = best_point['accuracy_mean']
ax2.plot(best_x, best_y, marker='*', markersize=15, color='gold', 
         markeredgecolor='black', markeredgewidth=2, label='Best performing')

ax2.axhline(y=best_y, color='gold', linestyle='--', alpha=0.7, linewidth=2)

ax2.set_xlabel('Number of rounds', fontweight='bold', fontsize=16)
ax2.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
ax2.tick_params(axis='both', labelsize=14)
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.legend(fontsize=12, loc='upper left', frameon=True, fancybox=True, shadow=True, 
           framealpha=1.0, facecolor='white', edgecolor='black')
apply_standard_plot_formatting(ax2, 'b', background_color=background_colors['white'], fontsize=20)
ax2.set_ylim(20, 45)
ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels([1, 2, 3])

ax3 = fig.add_subplot(gs[0, 2])

role_play_conditions = ['enable_role_play', 'disable_role_play']
role_play_labels = ['Yes', 'No']
x_pos = np.arange(len(role_play_conditions))

role_play_accuracies = []
role_play_stds = []

for condition in role_play_conditions:
    condition_data = role_play_summary[role_play_summary['exp_name'] == condition]
    if not condition_data.empty:
        role_play_accuracies.append(condition_data['accuracy_mean'].iloc[0])
        role_play_stds.append(condition_data['accuracy_std'].iloc[0])
    else:
        role_play_accuracies.append(0)
        role_play_stds.append(0)

bars = ax3.bar(x_pos, role_play_accuracies, 
            color=[colors['role_play']['enable_role_play'], colors['role_play']['disable_role_play']], 
            alpha=0.8, edgecolor='black', linewidth=1.5)

ax3.errorbar(x_pos, role_play_accuracies, yerr=role_play_stds, 
             fmt='none', color='black', alpha=0.5, capsize=3, capthick=1)

for i, (bar, acc) in enumerate(zip(bars, role_play_accuracies)):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

ax3.set_xlabel('Role Play', fontweight='bold', fontsize=16)
ax3.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
ax3.tick_params(axis='both', labelsize=14)
ax3.grid(True, alpha=0.2, axis='y', linewidth=0.8, linestyle='-')
apply_standard_plot_formatting(ax3, 'c', background_color=background_colors['white'], fontsize=20)
ax3.set_ylim(20, 45)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(role_play_labels)

ax4 = fig.add_subplot(gs[1, 0])

baseline_accuracy = summary_df[(summary_df['n_agents'] == 1) & 
                              (summary_df['n_rounds'] == 1) & 
                              (summary_df['has_search'] == False)]['accuracy_mean'].iloc[0]

components = ['Baseline', 'Multi-Agent', 'Evidence\nRetrieval', 'Iterative\nReasoning', 'Role Play', 'Discussion\nOrchestration', 'Final']
values = [baseline_accuracy]
cumulative = [baseline_accuracy]

add_agents_data = summary_df[(summary_df['n_agents'] == 3) & 
                            (summary_df['n_rounds'] == 1) & 
                            (summary_df['has_search'] == False)]
if not add_agents_data.empty:
    agents_improvement = add_agents_data['accuracy_mean'].iloc[0] - baseline_accuracy
    values.append(agents_improvement)
    cumulative.append(cumulative[-1] + agents_improvement)
else:
    values.append(0)
    cumulative.append(cumulative[-1])

add_search_data = summary_df[(summary_df['n_agents'] == 3) & 
                            (summary_df['n_rounds'] == 1) & 
                            (summary_df['has_search'] == True)]
if not add_search_data.empty:
    search_improvement = add_search_data['accuracy_mean'].iloc[0] - cumulative[-1]
    values.append(search_improvement)
    cumulative.append(cumulative[-1] + search_improvement)
else:
    values.append(0)
    cumulative.append(cumulative[-1])

add_rounds_data = summary_df[(summary_df['n_agents'] == 3) & 
                            (summary_df['n_rounds'] == 3) & 
                            (summary_df['has_search'] == True)]
if not add_rounds_data.empty:
    rounds_improvement = add_rounds_data['accuracy_mean'].iloc[0] - cumulative[-1]
    values.append(rounds_improvement)
    cumulative.append(cumulative[-1] + rounds_improvement)
else:
    values.append(0)
    cumulative.append(cumulative[-1])

enable_role_play_data = role_play_summary[role_play_summary['exp_name'] == 'enable_role_play']
disable_role_play_data = role_play_summary[role_play_summary['exp_name'] == 'disable_role_play']
if not enable_role_play_data.empty and not disable_role_play_data.empty:
    role_play_improvement = enable_role_play_data['accuracy_mean'].iloc[0] - disable_role_play_data['accuracy_mean'].iloc[0]
    values.append(role_play_improvement)
    cumulative.append(cumulative[-1] + role_play_improvement)
else:
    values.append(0)
    cumulative.append(cumulative[-1])

orchestration_best_data = orchestration_summary[orchestration_summary['exp_name'] == 'group_chat_with_orchestrator']
orchestration_worst_data = orchestration_summary[orchestration_summary['exp_name'] == 'independent']
if not orchestration_best_data.empty and not orchestration_worst_data.empty:
    orchestration_improvement = orchestration_best_data['accuracy_mean'].iloc[0] - orchestration_worst_data['accuracy_mean'].iloc[0]
    values.append(orchestration_improvement)
    cumulative.append(cumulative[-1] + orchestration_improvement)
else:
    values.append(0)
    cumulative.append(cumulative[-1])

values.append(cumulative[-1])

x_pos = np.arange(len(components))
component_colors = colors['component_breakdown']
bar_colors = [component_colors['Baseline'] if i == 0 
              else component_colors['Multi-Agent'] if i == 1
              else component_colors['Evidence_Retrieval'] if i == 2
              else component_colors['Iterative_Reasoning'] if i == 3
              else component_colors['Role_Play'] if i == 4
              else component_colors['Discussion_Orchestration'] if i == 5
              else component_colors['Final'] for i in range(len(components))]

bars = []
for i, (component, value) in enumerate(zip(components, values)):
    if i == 0 or i == len(components) - 1:
        bar = ax4.bar(i, value, bottom=0, color=bar_colors[i], alpha=0.8, 
                     edgecolor='black', linewidth=1.5, width=0.7)
        bars.append(bar)
    else:
        bottom = cumulative[i-1] if value > 0 else cumulative[i]
        height = abs(value)
        bar = ax4.bar(i, height, bottom=bottom, color=bar_colors[i], alpha=0.8,
                     edgecolor='black', linewidth=1.5, width=0.7)
        bars.append(bar)

for i in range(1, len(cumulative)):
    if i < len(cumulative) - 1:
        ax4.plot([i-0.35, i+0.35], [cumulative[i-1], cumulative[i-1]], 
                'k--', alpha=0.6, linewidth=1.2)

for i, (bar_group, value, cum_val) in enumerate(zip(bars, values, cumulative)):
    if isinstance(bar_group, list):
        bar = bar_group[0]
    else:
        bar = bar_group[0]
    
    if i == 0 or i == len(components) - 1:
        y_pos = bar.get_height() + 0.8
        text_val = cum_val
    else:
        y_pos = bar.get_y() + bar.get_height() + 0.8
        text_val = value
    
    ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
             f'{text_val:.1f}%' if i == 0 or i == len(components)-1 else f'+{text_val:.1f}%' if text_val > 0 else f'{text_val:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

total_improvement = cumulative[-1] - cumulative[0]
ax4.text(0.02, 0.98, f'Total Improvement: +{total_improvement:.1f}%', 
         transform=ax4.transAxes, fontsize=13, fontweight='bold',
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                  alpha=0.9, edgecolor='black', linewidth=1))

ax4.set_xlabel('System Components', fontweight='bold', fontsize=16)
ax4.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
ax4.tick_params(axis='both', labelsize=14)
ax4.grid(True, alpha=0.3, axis='y', linewidth=0.5, linestyle='-')
apply_standard_plot_formatting(ax4, 'd', background_color=background_colors['white'], fontsize=20)

ax4.set_ylim(20, max(cumulative) + 5)

ax4.set_xlim(-0.5, len(values) - 0.5)
ax4.set_xticks(range(len(values)))
ax4.set_xticklabels(components, fontsize=14, ha='center', rotation=0)

for spine in ax4.spines.values():
    spine.set_linewidth(1.2)

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
    colors['orchestration']['group_chat_with_orchestrator'],
    colors['orchestration']['group_chat_voting_only'],
    colors['orchestration']['independent'],
    colors['orchestration']['one_on_one_sync']
]

bars = ax5.bar(x_pos, orchestration_accuracies, 
               color=orchestration_colors, 
               alpha=0.8, edgecolor='black', linewidth=1.5)
ax5.errorbar(x_pos, orchestration_accuracies, yerr=orchestration_stds, 
             fmt='none', color='black', alpha=0.5, capsize=3, capthick=2)

for i, (bar, acc) in enumerate(zip(bars, orchestration_accuracies)):
    ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

ax5.set_xlabel('Discussion Strategy', fontweight='bold', fontsize=16)
ax5.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
ax5.tick_params(axis='both', labelsize=14)
ax5.grid(True, alpha=0.2, axis='y', linewidth=0.8, linestyle='-')
apply_standard_plot_formatting(ax5, 'e', background_color=background_colors['white'], fontsize=20)
ax5.set_ylim(20, 45)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(orchestration_labels, fontsize=13)

ax5_time = ax5.twinx()

time_line = ax5_time.plot(x_pos, orchestration_times, 
                         marker='o', linewidth=2.5, markersize=8,
                         color=colors['metrics']['time'], alpha=0.7,
                         markeredgecolor='black', markeredgewidth=1.5, 
                         linestyle='--', label='Processing Time')

ax5_time.errorbar(x_pos, orchestration_times, yerr=orchestration_time_stds, 
                 fmt='none', color=colors['metrics']['time'], alpha=0.7, 
                 capsize=3, capthick=2)

ax5_time.set_ylabel('Processing Time (s)', fontweight='bold', fontsize=16, color='black')
ax5_time.tick_params(axis='y', labelcolor='black', labelsize=13)
ax5_time.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=True, 
                framealpha=1.0, facecolor='white', edgecolor='black')

best_orchestration_idx = orchestration_accuracies.index(max(orchestration_accuracies)) if orchestration_accuracies else 0
if orchestration_accuracies:
    ax5.plot(best_orchestration_idx, orchestration_accuracies[best_orchestration_idx], 
             marker='*', markersize=15, color='gold', 
             markeredgecolor='black', markeredgewidth=2, 
             zorder=10, label='Best performing')
    ax5.legend(fontsize=12, loc='upper left', frameon=True, fancybox=True, shadow=True, 
               framealpha=1.0, facecolor='white', edgecolor='black')

expert_profiles_df = pd.read_csv('expert_profiles.csv')

def extract_official_specialty(job_title):
    """
    Extract medical specialty from job title using the official specialties list.
    
    This function performs comprehensive matching including exact matches, partial matches,
    common abbreviations, and specialty-specific variations to accurately categorize
    medical professionals into their respective specialties.
    
    Args:
        job_title (str): The job title to analyze
        
    Returns:
        str: The matched medical specialty or "Other/Unspecified" if no match found
    """
    if pd.isna(job_title):
        return "Unknown"
    
    job_title = str(job_title).lower().strip()
    
    specialty_mappings = {
        'Internal Medicine': ['internal', 'general medicine', 'internist', 'general practitioner', 'gp'],
        'Cardiology': ['cardiac', 'heart', 'cardiovascular', 'cardiologist'],
        'Neurology': ['neuro', 'neurologist', 'brain', 'nervous system'],
        'Emergency Medicine': ['emergency', 'er', 'acute care', 'trauma', 'urgent care'],
        'Infectious Disease': ['infectious', 'infection', 'id', 'communicable disease', 'tropical medicine'],
        'Pediatrics': ['pediatric', 'child', 'infant', 'neonatal', 'adolescent medicine'],
        'Endocrinology': ['endocrine', 'hormone', 'diabetes', 'thyroid', 'metabolic'],
        'Pharmacology': ['pharmacy', 'drug', 'medication', 'pharmaceutical', 'clinical pharmacy'],
        'Pulmonology': ['pulmonary', 'lung', 'respiratory', 'chest medicine', 'breathing'],
        'Gastroenterology': ['gastro', 'gi', 'digestive', 'stomach', 'liver', 'hepatology'],
        'Hematology': ['blood', 'hematologic', 'coagulation', 'bleeding disorders'],
        'Radiology': ['imaging', 'x-ray', 'mri', 'ct', 'ultrasound', 'diagnostic imaging'],
        'Nephrology': ['kidney', 'renal', 'dialysis', 'transplant nephrology'],
        'Psychiatry': ['mental health', 'psychological', 'behavioral health', 'psychiatric'],
        'Oncology': ['cancer', 'tumor', 'chemotherapy', 'radiation oncology', 'hematology-oncology'],
        'Genetics': ['genetic', 'genomic', 'hereditary', 'molecular genetics', 'clinical genetics'],
        'Dermatology': ['skin', 'dermatologic', 'cosmetic', 'aesthetic medicine'],
        'Pathology': ['laboratory', 'lab', 'anatomic pathology', 'forensic pathology'],
        'Geriatrics': ['elderly', 'aging', 'senior', 'gerontology'],
        'Immunology': ['immune', 'autoimmune', 'immunologic'],
        'Rheumatology': ['arthritis', 'joint', 'autoimmune', 'connective tissue'],
        'Urology': ['urologic', 'kidney', 'bladder', 'prostate', 'genitourinary'],
        'Obstetrics and Gynecology': ['obstetrics', 'gynecology', 'ob/gyn', 'obgyn', 'womens health', 'maternal'],
        'Surgery': ['surgical', 'surgeon', 'operative', 'general surgery'],
        'Preventive Medicine': ['prevention', 'public health', 'occupational health', 'community medicine'],
        'Critical Care Medicine': ['critical care', 'intensive care', 'icu', 'ccm', 'critical'],
        'Toxicology': ['poison', 'toxic', 'environmental health', 'occupational toxicology'],
        'Anesthesiology': ['anesthesia', 'pain management', 'perioperative', 'anesthetist'],
        'Neurosurgery': ['brain surgery', 'spine surgery', 'neurosurgeon', 'cranial'],
        'Family Medicine': ['family practice', 'primary care', 'family physician', 'community medicine'],
        'Vascular Medicine': ['vascular', 'angiology', 'blood vessels', 'circulation'],
        'Ophthalmology': ['eye', 'vision', 'retina', 'glaucoma', 'cataract'],
        'Orthopedics': ['bone', 'joint', 'musculoskeletal', 'sports medicine', 'fracture'],
        'Occupational Medicine': ['workplace', 'industrial', 'occupational health', 'work-related'],
        'Sports Medicine': ['athletic', 'exercise', 'fitness', 'sports injury'],
        'Public Health': ['epidemiology', 'population health', 'community health', 'health policy'],
        'Clinical Research': ['research', 'clinical trial', 'biomedical research', 'translational'],
        'Sleep Medicine': ['sleep', 'sleep disorders', 'sleep study', 'insomnia'],
        'Allergy and Immunology': ['allergy', 'allergic', 'asthma', 'immunologic'],
        'Biostatistics': ['statistics', 'data analysis', 'epidemiologic', 'biostatistical'],
        'Medical Ethics': ['ethics', 'bioethics', 'medical law', 'healthcare ethics'],
        'Neonatology': ['newborn', 'nicu', 'premature', 'neonatal intensive care'],
        'Nutrition': ['dietitian', 'nutritionist', 'clinical nutrition', 'dietary'],
        'Epidemiology': ['disease surveillance', 'outbreak', 'population studies', 'epidemiologic'],
        'Rehabilitation Medicine': ['rehabilitation', 'rehab', 'physical medicine', 'disability'],
        'Sexual Health': ['sexual', 'reproductive health', 'std', 'sexual dysfunction'],
        'Reproductive Medicine': ['fertility', 'infertility', 'ivf', 'reproductive endocrinology'],
        'Transplant Medicine': ['transplant', 'organ donation', 'transplantation'],
        'Clinical Pathology': ['clinical lab', 'laboratory medicine', 'diagnostic pathology']
    }
    
    for specialty in specialty_mappings:
        specialty_lower = specialty.lower()
        if specialty_lower in job_title:
            return specialty
        elif specialty_lower in specialty_mappings:
            for variant in specialty_mappings[specialty]:
                if variant in job_title:
                    return specialty
    
    common_medical_terms = ['doctor', 'physician', 'md', 'do', 'nurse', 'practitioner', 'specialist', 'consultant']
    if any(term in job_title for term in common_medical_terms):
        for specialty in specialty_mappings:
            specialty_words = specialty.lower().split()
            if any(word in job_title for word in specialty_words if len(word) > 3):
                return specialty
    return "Other/Unspecified"

expert_profiles_df['official_specialty'] = expert_profiles_df['job_title'].apply(extract_official_specialty)
ax_role_play = fig.add_subplot(gs[2, :2])

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

datasets = expert_profiles_df['dataset'].unique()
specialties = expert_profiles_df['official_specialty'].value_counts()
specialties = specialties[specialties.index != "Other/Unspecified"]
top_specialties = specialties.head(12).index

specialty_by_dataset = {}
for dataset in datasets:
    dataset_df = expert_profiles_df[expert_profiles_df['dataset'] == dataset]
    specialty_counts = dataset_df['official_specialty'].value_counts()
    specialty_by_dataset[dataset] = specialty_counts

dataset_colors, fallback_colors = get_dataset_colors(theme='manchester_united_official')
dataset_color_map = {}
for i, dataset in enumerate(datasets):
    if dataset in dataset_colors:
        dataset_color_map[dataset] = dataset_colors[dataset]
    else:
        fallback_idx = i % len(fallback_colors)
        dataset_color_map[dataset] = fallback_colors[fallback_idx]

specialty_colors = {}
for i, specialty in enumerate(top_specialties):
    color_idx = i % len(dataset_colors)
    specialty_colors[specialty] = list(dataset_colors.values())[color_idx]
specialty_colors['Others'] = fallback_colors[0] if fallback_colors else colors['metrics']['cost']

dataset_data = {}
for dataset in datasets:
    dataset_data[dataset] = []
    total_experts = len(expert_profiles_df[(expert_profiles_df['dataset'] == dataset) & (expert_profiles_df['official_specialty'] != "Other/Unspecified")])
    
    for specialty in top_specialties:
        specialty_count = specialty_by_dataset[dataset].get(specialty, 0)
        percentage = (specialty_count / total_experts) * 100 if total_experts > 0 else 0
        dataset_data[dataset].append(percentage)
    
    others_count = sum(specialty_by_dataset[dataset].get(spec, 0) 
                      for spec in specialty_by_dataset[dataset].index 
                      if spec not in top_specialties and spec != "Other/Unspecified")
    others_percentage = (others_count / total_experts) * 100 if total_experts > 0 else 0
    dataset_data[dataset].append(others_percentage)

x_positions = np.arange(len(datasets))
bottom_values = np.zeros(len(datasets))

specialty_labels = list(top_specialties) + ['Others']

for i, specialty in enumerate(specialty_labels):
    specialty_percentages = [dataset_data[dataset][i] for dataset in datasets]
    
    bars = ax_role_play.bar(x_positions, specialty_percentages, 
                           bottom=bottom_values,
                           color=specialty_colors[specialty], alpha=0.8, 
                           edgecolor='black', linewidth=1.5, 
                           label=specialty)
    
    bottom_values += specialty_percentages

ax_role_play.set_xticks(x_positions)
ax_role_play.set_xticklabels([dataset_mapping[dataset] for dataset in datasets], 
                            fontsize=13)
ax_role_play.set_ylabel('Percentage of Experts (%)', fontweight='bold', fontsize=16, color='black')
ax_role_play.set_xlabel('Medical Datasets', fontweight='bold', fontsize=16, color='black')
ax_role_play.tick_params(axis='both', labelsize=13, colors='black')
ax_role_play.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
apply_standard_plot_formatting(ax_role_play, 'f', background_color=background_colors['white'], fontsize=20)

ax_role_play.legend(loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.1), fontsize=12, frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('agent_configuration_analysis.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()