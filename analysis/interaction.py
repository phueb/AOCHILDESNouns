import matplotlib.pyplot as plt
import pandas as pd

from abstractfirst.figs import make_line_fig
from abstractfirst import configs

IV = 'normalize_cols'
DV = 's1/sum(s)'
YLIMS = [0, 0.5]

df = pd.read_csv(configs.Dirs.results / 'results.csv')
if not len(df) == 8:
    raise ValueError(f'This script is designed to analyze the interaction of 3 IVs only.'
                     f'Expected 2^3=8 rows, but found {len(df)} rows in df.')

for level in df[IV].unique():
    description = f'{IV}={level}'
    print(description)
    df_level = df[df[IV] == level]

    # make 2 rows, one for each age & make 2 cols, one for each targets_control
    unstacked = df_level[['targets_control', 'age', DV]].set_index(['targets_control', 'age']).unstack(0)
    unstacked.columns = unstacked.columns.droplevel()

    # plot
    label2y = {'nouns': unstacked[False].to_numpy(),      # targets_control = False
               'non-nouns': unstacked[True].to_numpy(),   # targets_control = True
               }
    ages = [a for a in unstacked.index]
    fig = make_line_fig(label2y,
                        title=description,
                        x_axis_label='Age Range',
                        y_axis_label='Proportion of Variance Explained by s1',
                        x_ticks=[0, 1],
                        x_tick_labels=ages,
                        y_lims=YLIMS,
                        )
    plt.show()
