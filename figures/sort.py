import pandas as pd

df = pd.read_csv('main_comparison.csv')

df = df.sort_values(['model', 'dataset', 'exp_name'])

print(df['exp_name'].unique())
df.to_csv('main_comparison.csv', index=False)