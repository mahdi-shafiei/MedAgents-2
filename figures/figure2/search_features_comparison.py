"""Search Features Comparison Plot - Figure 2b"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    from .plot_utils import get_manchester_colors, get_figure_2_colors, apply_medagents_style
except ImportError:
    from plot_utils import get_manchester_colors, get_figure_2_colors, apply_medagents_style

def plot_search_features_comparison(ax, df, colors, panel_label='B'):
    """Compare search features ablation"""
    
    # Filter data for search features ablation
    features_df = df[df['ablation'] == 'search_features']
    
    feature_mapping = {
        'baseline': 'Baseline',
        'no_document_review': 'No Doc Review',
        'no_query_rewrite': 'No Query Rewrite', 
        'no_rewrite_no_review': 'No Both'
    }
    
    if not features_df.empty:
        features_df['feature_name'] = features_df['exp_name'].map(feature_mapping)
        
        # Get colors for each feature
        colors_list = [colors['features'].get(exp_name, colors['metrics']['accuracy']) 
                      for exp_name in features_df['exp_name']]
        
        bars = ax.bar(range(len(features_df)), features_df['accuracy'], 
                     color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value annotations
        for i, (bar, row) in enumerate(zip(bars, features_df.itertuples())):
            height = bar.get_height()
            ax.annotate(f'{row.accuracy:.0f}%', 
                       (bar.get_x() + bar.get_width()/2., height + 0.5),
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Show cost in center
            ax.annotate(f'{row.avg_cost:.1f}¢', 
                       (bar.get_x() + bar.get_width()/2., height/2),
                       ha='center', va='center', fontsize=9, fontweight='bold', color='white')

        ax.set_xticks(range(len(features_df)))
        ax.set_xticklabels(features_df['feature_name'], fontsize=10, rotation=45, ha='right')
        
    else:
        # Sample data
        features = ['Baseline', 'No Doc Review', 'No Query Rewrite', 'No Both']
        accuracies = [33, 31, 27, 29]
        costs = [3.72, 1.74, 3.73, 1.0]
        colors_list = [colors['features']['baseline'], colors['features']['no_document_review'],
                      colors['features']['no_query_rewrite'], colors['features']['no_rewrite_no_review']]
        
        bars = ax.bar(range(len(features)), accuracies, color=colors_list, alpha=0.8, 
                     edgecolor='black', linewidth=2)
        
        # Add value annotations
        for i, (bar, acc, cost) in enumerate(zip(bars, accuracies, costs)):
            height = bar.get_height()
            ax.annotate(f'{acc:.0f}%', 
                       (bar.get_x() + bar.get_width()/2., height + 0.5),
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.annotate(f'{cost:.1f}¢', 
                       (bar.get_x() + bar.get_width()/2., height/2),
                       ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, fontsize=10, rotation=45, ha='right')

    # Styling
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=11)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax.set_ylim(20, 40)
    
    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=15, fontweight='bold')
    
    return ax

def main():
    """Example usage when run as standalone script"""
    apply_medagents_style()
    colors = get_figure_2_colors()
    
    # Load real data
    try:
        df = pd.read_csv('../search_ablation.csv')
        print(f"Loaded {len(df)} search ablation records from real data")
    except FileNotFoundError:
        print("Warning: search_ablation.csv not found, creating sample data")
        df = pd.DataFrame({
            'ablation': ['search_features', 'search_features', 'search_features', 'search_features'],
            'exp_name': ['baseline', 'no_document_review', 'no_query_rewrite', 'no_rewrite_no_review'],
            'accuracy': [33, 31, 27, 29],
            'avg_time': [461, 376, 548, 291],
            'avg_cost': [3.72, 1.74, 3.73, 1.0]
        })
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_search_features_comparison(ax, df, colors, panel_label='B')
    plt.tight_layout()
    plt.savefig('search_features_comparison_example.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()