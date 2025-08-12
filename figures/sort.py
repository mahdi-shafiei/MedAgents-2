import pandas as pd

df = pd.read_csv('main_comparison.csv')

df = df.sort_values(['method', 'model', 'dataset'], ascending=[True, True, True])

df.to_csv('main_comparison.csv', index=False)