"""Overall Performance Bar Chart - Figure 0a"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import get_manchester_colors, get_figure_0_colors, apply_medagents_style

def plot_overall_performance_bar(ax, df, colors, panel_label):
    """Create overall performance bar chart with time series overlay"""
    
    # Configuration
    MODELS = ['o3-mini', 'gpt-4o', 'gpt-4o-mini']
    METHODS = ['CoT', 'CoT-SC', 'MedPrompt', 'MultiPersona', 'MedAgents', 'AFlow', 
               'MedAgents-2', 'Few-shot', 'MDAgents', 'SPO', 'Self-refine', 'MedRAG', 'Zero-shot']
    
    method_colors = colors.get('methods', {})
    
    # Calculate method statistics
    method_stats = {}
    for method in METHODS:
        method_df = df[df['method'] == method]
        if not method_df.empty:
            model_accuracies = {}
            model_times = {}
            for model in MODELS:
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

    # Sort methods by overall performance
    overall_means = []
    for method in METHODS:
        if method in method_stats:
            all_values = []
            for model in MODELS:
                if model in method_stats[method]['accuracy']:
                    all_values.extend(method_stats[method]['accuracy'][model]['values'])
            if all_values:
                overall_means.append((method, np.mean(all_values)))

    overall_means.sort(key=lambda x: x[1], reverse=True)
    methods_sorted = [item[0] for item in overall_means]

    # Plot bars
    bar_width = 0.3
    x_positions = np.arange(len(methods_sorted))
    alphas = [0.9, 0.7, 0.5]

    for i, model in enumerate(MODELS):
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
        
        # Bars
        ax.bar(x_pos, model_means, bar_width,
               color=[method_colors.get(method, 'gray') for method in methods_sorted],
               alpha=alphas[i], edgecolor='black', linewidth=1.5, label=f'{model}')
        
        # Error bars
        ax.errorbar(x_pos, model_means, yerr=model_stds, fmt='none',
                   ecolor='black', capsize=4, capthick=2, linewidth=2, alpha=0.8)
        
        # Data points
        for j, (x, mean, values) in enumerate(zip(x_pos, model_means, model_values_list)):
            if len(values) > 0:
                for value in values:
                    ax.plot(x, value, 'o', color='darkred' if i == 0 else 'darkblue', 
                           markersize=4, alpha=0.7, markeredgecolor='black', markeredgewidth=0.5)

    # Time series overlay
    ax_twin = ax.twinx()
    for i, model in enumerate(MODELS):
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
        
        ax_twin.plot(x_positions, model_time_means, color=line_color, linestyle=line_style,
                    marker=marker_style, markersize=10, linewidth=4, alpha=0.9,
                    label=f'{model} (Time)', markeredgecolor='white', markeredgewidth=1)
        
        ax_twin.errorbar(x_positions, model_time_means, yerr=model_time_stds, fmt='none',
                        ecolor=line_color, capsize=3, capthick=2, linewidth=2, alpha=0.7)

    # Best method star
    if methods_sorted:
        best_method_idx = 0
        best_method = methods_sorted[0]
        best_x = x_positions[best_method_idx]
        best_y = max([method_stats[best_method]['accuracy'][model]['mean'] 
                     for model in MODELS if model in method_stats[best_method]['accuracy']])

        ax.plot(best_x, best_y + 3, marker='*', markersize=25, color='gold', 
               markeredgecolor='black', markeredgewidth=3, zorder=15, label='Best Overall')

    # Styling
    ax.set_ylim(8, 48)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=18)
    ax_twin.set_ylabel('Execution Time (seconds)', fontweight='bold', fontsize=18)
    ax.set_xlabel('Methods (Ranked by Performance)', fontweight='bold', fontsize=18)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods_sorted, fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax_twin.tick_params(axis='both', labelsize=15)
    ax.grid(True, alpha=0.4, axis='y', linewidth=0.8, linestyle=':')
    
    # Legend
    model_legend_elements = []
    for i, model in enumerate(MODELS):
        model_legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=alphas[i],
                                                 edgecolor='black', linewidth=1.5, label=f'{model} (Accuracy)'))
    
    time_legend_elements = []
    for i, model in enumerate(MODELS):
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
    ax.legend(handles=combined_legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
             title='Legend', fontsize=13, ncol=1, frameon=True, fancybox=True, shadow=True, 
             framealpha=0.95, facecolor='white', edgecolor='black', title_fontsize=15)
    
    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=25, fontweight='bold')
    
    return ax, ax_twin

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
        models = ['gpt-4o', 'gpt-4o-mini'] * 3
        methods = ['MedAgents-2', 'MedAgents-2', 'CoT', 'CoT', 'Zero-shot', 'Zero-shot']
        df = pd.DataFrame({
            'method': methods,
            'model': models,
            'accuracy': [35.2, 33.8, 28.1, 26.5, 18.5, 16.2],
            'avg_time': [65.3, 58.2, 42.1, 38.7, 15.2, 13.8],
            'run_id': [0, 1, 0, 1, 0, 1]
        })
    
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_overall_performance_bar(ax, df, colors, panel_label='A')
    plt.tight_layout()
    plt.savefig('overall_performance_bar_example.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()