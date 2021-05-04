import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from itertools import product

from aochildesnouns import configs

df = pd.read_csv(configs.Dirs.results / 'results.csv')

FIG_SIZE = (6, 8)
NUM_ROWS, NUM_COLS = 8, 2  # num rows are for all factor combinations except age, word list, and normalization
WIDTH = 0.1
X_TICK_LABELS = ['nouns', 'non-nouns']
Y_LIMS = [0.5, 1.0]

# add empty axis to make space for legend
fig, ax_mat = plt.subplots(NUM_ROWS + 1, NUM_COLS, figsize=FIG_SIZE, dpi=configs.Fig.dpi)
plt.suptitle('Fragmentation', fontsize=16)

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

for row_id, ax_row in enumerate(ax_mat):

    if row_id == NUM_ROWS:  # make last axis empty for legend
        for ax in ax_row:
            ax.axis('off')
        break

    # each row represent a combination of 3 factors
    levels = next(factor_combinations)
    print(levels)

    for ax, ax_title in zip(ax_row, ['raw frequency', 'normalized']):

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
        print()

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if row_id == 0:
            ax.set_title(ax_title)

        ax.set_yticks([0.5, 0.75, 1.0])
        ax.set_yticklabels([0.5, 0.75, 1.0], fontsize=6)
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
# plt.tight_layout()
plt.subplots_adjust(hspace=-0.4)
plt.show()
