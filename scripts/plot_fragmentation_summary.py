import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from itertools import product
from pathlib import Path
import numpy as np

df = pd.read_csv(Path(__file__).parent.parent / 'results' / 'results.csv')

FIG_SIZE = (6, 4)
NUM_ROWS, NUM_COLS = 2, 2  # (lemmatization, direction)
WIDTH = 0.1
Y_LIMS1 = [0.7, 1.01]
Y_LIMS2 = [0.6, 0.91]

X_TICK_LABELS = ['nouns', 'non-nouns']

# add empty row axis to make space for legend.
# add empty col axis to make space for labels for conditions
fig, ax_mat = plt.subplots(NUM_ROWS + 1, NUM_COLS + 1,
                           figsize=FIG_SIZE,
                           dpi=192,
                           constrained_layout=True)
plt.suptitle('Fragmentation', fontsize=16)

f2f = {
    'direction': 'direction',
    'lemmas': 'lemmatization',
    'punctuation': 'punctuation',
}

l2l = {
    'r': 'forward',
    'l': 'backward',
    'lemmas': 'lemmatization',
    'keep': 'intact',
    'remove': 'removed',
    False: 'False',
    True: 'True',
}

direction2direction = {
    'backward': 'l',
    'forward': 'r',
}

num_age_groups = 2

factors = [
    'lemmas',
]

factor_levels = [
    [False, True],
]
factor_combinations = product(*factor_levels)

for row_id, ax_row in enumerate(ax_mat):

    if row_id == NUM_ROWS:  # make last axis empty for legend
        for ax in ax_row:
            ax.axis('off')
        break

    # each row represent a combination of 3 factors
    levels = next(factor_combinations)
    print(levels)

    ax_row[0].axis('off')
    ax_row[0].text(x=0.0,
                   y=0.0,
                   s='\n'.join([f'{f2f[f]}={l2l[l]}' for f, l in zip(factors, levels)]),
                   )

    for ax, direction in zip(ax_row[1:], ['backward', 'forward']):

        # get data
        cond = (df[factors[0]] == levels[0]) & \
               (df['punctuation'] == 'keep') & \
               (df['normalize_cols'] == False) & \
               (df['direction'] == direction2direction[direction])

        df_ax = df.where(cond).dropna()
        y = df_ax['frag'].values
        print(y)
        assert len(y) == 4

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if row_id == 0:
            ax.set_title(f'direction={direction}', fontsize=10)

        y_ticks = [0.6, 0.7, 0.8, 0.9, 1.0]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks, fontsize=6)

        if direction == 'l':
            ax.set_ylim(Y_LIMS1)
        else:
            ax.set_ylim(Y_LIMS2)

        if row_id == NUM_ROWS - 1:
            ax.set_xticks([1, 2])
            ax.set_xticklabels(X_TICK_LABELS)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

        ax.grid(axis='y')

        # plot 2 groups of 2 bars: both age groups, and raw vs. normalized data
        x = [1 - WIDTH / 2, 1 + WIDTH / 2, 2 - WIDTH / 2, 2 + WIDTH / 2]
        ax.bar(x,
               y,
               color=['C0', 'C1', 'C0', 'C1'],
               width=WIDTH,
               zorder=3
               )


legend_elements = [Patch(facecolor='C0',
                         label='age group 1'),
                   Patch(facecolor='C1',
                         label='age group 2')
                   ]
fig.legend(handles=legend_elements,
           prop={'size': 12},
           bbox_to_anchor=(0.5, 0.2),  # distance from bottom-left
           loc='upper center',
           frameon=False)

# TODO this script must be run in terminal so that subplots are close to each other

# fig.set_constrained_layout_pads(h_pad=0, hspace=500, wspace=330)
plt.show()
