import pandas as pd

from aochildesnouns import configs

df = pd.read_csv(configs.Dirs.results / 'results.csv')

col_names = [
    'age',
    'direction',
    'lemmas',
    'punctuation',
    'x-types',
    'y-types',
    'x-tokens',
]

print('targets_control == False')
print(df[df['targets_control'] == False].to_latex(index=False, columns=col_names))

print('targets_control == True')
print(df[df['targets_control'] == True].to_latex(index=False, columns=col_names))
