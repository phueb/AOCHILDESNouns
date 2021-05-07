import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from itertools import product
from pathlib import Path
import numpy as np

df = pd.read_csv(Path(__file__).parent.parent / 'results' / 'results.csv')

FIG_SIZE = (6, 8)
NUM_ROWS, NUM_COLS = 8, 2  # num rows are for all factor combinations except age, word list, and normalization
WIDTH = 0.1
X_TICK_LABELS = ['nouns', 'non-nouns']
Y_LIMS = [0.6, 1.01]

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

num_age_groups = 2

factors = [
    'direction',
    'lemmas',
    'punctuation',
]

factor_levels = [
    ['l', 'r'],
    [False, True],
    ['keep', 'remove'],
]
factor_combinations = product(*factor_levels)

differences1 = []
differences2 = []
differences3 = []
differences4 = []

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

    for ax, ax_title in zip(ax_row[1:], ['raw frequency', 'normalized']):

        normalize_cols = True if ax_title == 'normalized' else False

        # get data
        cond = (df[factors[0]] == levels[0]) & \
               (df[factors[1]] == levels[1]) & \
               (df[factors[2]] == levels[2]) & \
               (df['normalize_cols'] == normalize_cols)

        df_ax = df.where(cond).dropna()
        y = df_ax['frag'].values
        assert len(y) == 4
        print(y)

        # collect stats
        if ax_title == 'raw frequency':
            differences1.append(round(abs(y[0] - y[1]), 3))
            differences2.append(round(abs(y[2] - y[3]), 3))
        else:
            differences3.append(round(abs(y[0] - y[1]), 3))
            differences4.append(round(abs(y[2] - y[3]), 3))

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if row_id == 0:
            ax.set_title(ax_title)

        y_ticks = [0.6, 0.7, 0.8, 0.9, 1.0]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks, fontsize=6)
        ax.set_ylim(Y_LIMS)

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
           bbox_to_anchor=(0.5, 0.08),  # distance from bottom-left
           loc='upper center',
           frameon=False)

# TODO this script must be run in terminal so that subplots are close to each other

# fig.set_constrained_layout_pads(h_pad=0, hspace=500, wspace=330)
plt.show()


print(differences1)
print(np.mean(differences1))
print(np.std(differences1))
print()
print(differences2)
print(np.mean(differences2))
print(np.std(differences2))
print()


print(np.mean(differences3))
print(np.mean(differences4))