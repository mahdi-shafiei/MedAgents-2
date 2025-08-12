import pandas as pd
import numpy as np
from scipy import stats

# Read the data files
main_df = pd.read_csv('main_comparison.csv')
agent_df = pd.read_csv('agent_configuration.csv')
search_df = pd.read_csv('search_ablation.csv')
role_play_df = pd.read_csv('role_play.csv')
orchestration_df = pd.read_csv('orchestration_style.csv')

print("\nFigure 1 Key Metrics:")
# Analyze main comparison data
methods = main_df['exp_name'].unique()
datasets = main_df['dataset'].unique()

# Overall performance metrics
best_method = main_df.groupby('exp_name')['accuracy'].mean().idxmax()
best_accuracy = main_df[main_df['exp_name'] == best_method]['accuracy'].mean()
best_std = main_df[main_df['exp_name'] == best_method]['accuracy'].std()

# Dataset-specific performance
dataset_performance = main_df.groupby('dataset')['accuracy'].agg(['mean', 'std', 'max'])
best_dataset = dataset_performance['max'].idxmax()
dataset_improvement = ((dataset_performance['max'] - dataset_performance['mean']) / dataset_performance['mean'] * 100)

# Cost-effectiveness analysis
cost_metrics = main_df.groupby('exp_name').agg({
    'accuracy': 'mean',
    'avg_cost': 'mean',
    'avg_time': 'mean'
}).reset_index()
cost_efficiency = cost_metrics['accuracy'] / cost_metrics['avg_cost']
most_efficient = cost_metrics.loc[cost_efficiency.idxmax(), 'exp_name']

print(f"- Best performing method: {best_method} (accuracy: {best_accuracy:.1f}% ± {best_std:.1f}%)")
print(f"- Performance range: {main_df['accuracy'].min():.1f}% - {main_df['accuracy'].max():.1f}%")
print(f"- Most cost-efficient method: {most_efficient}")
print(f"- Number of datasets evaluated: {len(datasets)}")
print(f"- Best performing dataset: {best_dataset} (improvement: {dataset_improvement[best_dataset]:.1f}%)")
print(f"- Average query cost range: {cost_metrics['avg_cost'].min():.2f}-{cost_metrics['avg_cost'].max():.2f} cents")
print(f"- Average processing time: {cost_metrics['avg_time'].mean():.1f}s ± {cost_metrics['avg_time'].std():.1f}s")

print("\nFigure 2 Key Metrics:")
# Enhanced agent configuration analysis
def parse_exp_name(exp_name):
    parts = exp_name.split('_')
    n_agents = int(parts[0])
    n_rounds = int(parts[2])
    has_search = 'with_search' in exp_name
    return n_agents, n_rounds, has_search

agent_df[['n_agents', 'n_rounds', 'has_search']] = agent_df['exp_name'].apply(
    lambda x: pd.Series(parse_exp_name(x))
)

# Agent count analysis
agent_performance = agent_df.groupby('n_agents').agg({
    'accuracy': ['mean', 'std'],
    'avg_time': ['mean', 'std']
}).reset_index()
optimal_agents = agent_performance.loc[agent_performance[('accuracy', 'mean')].idxmax()]

# Round analysis
round_impact = agent_df.groupby('n_rounds').agg({
    'accuracy': ['mean', 'std'],
    'avg_time': ['mean', 'std']
}).reset_index()
optimal_rounds = round_impact.loc[round_impact[('accuracy', 'mean')].idxmax()]

# Role-play impact
role_play_impact = role_play_df.groupby('exp_name').agg({
    'accuracy': ['mean', 'std'],
    'avg_time': 'mean'
}).reset_index()
role_play_improvement = ((role_play_impact.loc[0, ('accuracy', 'mean')] - 
                         role_play_impact.loc[1, ('accuracy', 'mean')]) / 
                        role_play_impact.loc[1, ('accuracy', 'mean')] * 100)

print(f"- Optimal agent configuration: {int(optimal_agents['n_agents'])} agents")
print(f"- Peak accuracy: {optimal_agents[('accuracy', 'mean')]:.1f}% ± {optimal_agents[('accuracy', 'std')]:.1f}%")
print(f"- Optimal number of rounds: {int(optimal_rounds.name)} (accuracy: {optimal_rounds[('accuracy', 'mean')]:.1f}%)")
print(f"- Role-play improvement: {role_play_improvement:.1f}%")
print(f"- Processing time increase per agent: {agent_performance[('avg_time', 'mean')].diff().mean():.1f}s")

print("\nFigure 3 Key Metrics:")
# Enhanced search ablation analysis
modality_df = search_df[search_df['ablation'] == 'search_modality']
features_df = search_df[search_df['ablation'] == 'search_features']
history_df = search_df[search_df['ablation'] == 'search_history']

# Modality analysis
best_modality = modality_df.loc[modality_df['accuracy'].idxmax()]
modality_improvement = ((best_modality['accuracy'] - modality_df['accuracy'].min()) / 
                       modality_df['accuracy'].min() * 100)

# Feature impact
feature_impact = features_df.groupby('exp_name').agg({
    'accuracy': ['mean', 'std'],
    'avg_time': 'mean'
}).reset_index()
feature_contribution = feature_impact['accuracy']['mean'].pct_change() * 100

# History analysis
history_comparison = history_df.groupby('exp_name').agg({
    'accuracy': ['mean', 'std'],
    'avg_time': 'mean'
}).reset_index()

print(f"- Best search configuration: {best_modality['exp_name']} ({best_modality['accuracy']:.1f}% accuracy)")
print(f"- Improvement over baseline: {modality_improvement:.1f}%")
print(f"- Feature contribution range: {feature_contribution.min():.1f}% to {feature_contribution.max():.1f}%")
print(f"- Processing time range: {modality_df['avg_time'].min():.1f}s to {modality_df['avg_time'].max():.1f}s")
print(f"- Query cost reduction: {((modality_df['avg_cost'].max() - modality_df['avg_cost'].min()) / modality_df['avg_cost'].max() * 100):.1f}%")
print(f"- Shared history advantage: {((history_comparison.loc[0, ('accuracy', 'mean')] - history_comparison.loc[1, ('accuracy', 'mean')]) / history_comparison.loc[1, ('accuracy', 'mean')] * 100):.1f}%") 