
"""
compute expected difference in conditional entropy H(X|Y) of simulated co-occurrence matrices between age group 1 and 2
while accounting for differences in their shapes.

there are more columns in age group 2 than age group 1, which reduces estimate of H(X|Y) due to chance alone.

PROP_AGE_2 really matters. set this to high to simulate naturalistic co-occurrence data.
when this is low, then co-occurrences are randomly distributed,
and therefore age group differences due to chance increase in likelihood.

"""
import matplotlib.pyplot as plt
from pyitlib import discrete_random_variable as drv
import numpy as np


N = 80_000  # number of total co-occurrences
CO_MAT_SHAPE_1 = (600, 2000)
CO_MAT_SHAPE_2 = (630, 2600)
PROP_AGE_2 = 0.90  # proportion of co-occurrences that involve entropy-maximizing contexts
PROP_DIFFS = [0.00, 0.01, 0.02, 0.03]  # proportion of observations with entropy-maximizing contexts added to group 1
NUM_REPEAT = 5

x_ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [20, 30, 40, 50, 60, 70, 80, 90, 100]

for pd in PROP_DIFFS:
    assert pd < PROP_AGE_2


class Vocab:
    """
    a vocabulary where word frequency distributed as power-law
    """
    def __init__(self, num_rows, num_cols):
        self.x = np.arange(num_rows)
        self.y = np.arange(num_cols)

        p_x = np.array([1 / (i + 1) for i in range(num_rows)])
        p_y = np.array([1 / (i + 1) for i in range(num_cols)])

        self.p_x = p_x / p_x.sum()
        self.p_y = p_y / p_y.sum()


def compute_nce(prop_, vocab):
    """
    compute normalized conditional entropy, given:
    1) a vocab, and
    2) the proportion of observations (co-occurrences) that involve entropy-maximizing contexts

    normalization does not really have a purpose here - it does not de-bias due to shape differences

    """

    num_entropic_observations = int(N * prop_)
    num_remaining_observations = N - num_entropic_observations

    # get non-entropic observations (pseudo-Zipfian probability)
    x = np.random.choice(vocab.x, size=num_remaining_observations, p=vocab.p_x).tolist()
    y = np.random.choice(vocab.y, size=num_remaining_observations, p=vocab.p_y).tolist()

    # get entropic observations (uniform probability)
    x += np.random.choice(vocab.x, size=num_entropic_observations, p=vocab.p_x).tolist()
    y += np.random.choice(vocab.y[:x_tick], size=num_entropic_observations, p=None).tolist()

    # compute
    xy = np.vstack((x, y))
    res = drv.entropy_conditional(x, y) / drv.entropy_joint(xy)

    return res


vocab1 = Vocab(*CO_MAT_SHAPE_1)
vocab2 = Vocab(*CO_MAT_SHAPE_2)

prop_diff2xy_ce_diffs = {f: [] for f in PROP_DIFFS}
for prop_diff in prop_diff2xy_ce_diffs:

    for x_tick in x_ticks:

        # compute multiple times
        xy_ces1 = []
        xy_ces2 = []
        for _ in range(NUM_REPEAT):
            xy_ces1_i = compute_nce(PROP_AGE_2 + prop_diff, vocab1)
            xy_ces2_i = compute_nce(PROP_AGE_2, vocab2)
            # collect
            xy_ces1.append(xy_ces1_i)
            xy_ces2.append(xy_ces2_i)

        # compute difference
        avg_diff = np.mean(xy_ces1) - np.mean(xy_ces2)

        # collect
        prop_diff2xy_ce_diffs[prop_diff].append(avg_diff)

        # note:
        # this difference is always bigger (due to chance)
        # the point is to show that this difference increase as entropy-maximization increases

        # note:
        # prop_diff is the difference (between age groups) in prop of co-occurrences that involve e-maximizing contexts

        print(f'x_tick={x_tick:>6} prop={prop_diff:>6} difference={avg_diff}')


# fig
fig, ax = plt.subplots(1, figsize=(6, 4), dpi=300)
ax.set_ylabel('Normalized H(X|Y) Difference\n(Age 1 - Age 2)', fontsize=12)
ax.set_xlabel('Number of entropy-maximizing contexts', fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
x_tick_labels = [i for i in x_ticks if i >= 10 or i == 1]
ax.set_xticks(x_tick_labels)
ax.set_xticklabels(x_tick_labels)
ax.yaxis.grid(True)
# make sure y=0 is included in y-axis
max_y = round(prop_diff2xy_ce_diffs[PROP_DIFFS[-1]][0], 2)
# ax.set_ylim([0, max_y])
# plot expected difference in conditional entropy as function of proportion
color_ids = iter(range(2, 10))
for prop_diff, xy_ce_diff in prop_diff2xy_ce_diffs.items():
    ax.plot(x_ticks,
            xy_ce_diff,
            color=f'C{next(color_ids)}',  # avoid orange and blue colors
            label=r'$\rho={{{}}}$'.format(prop_diff),
            )
plt.legend(frameon=False,
           ncol=len(PROP_DIFFS),
           bbox_to_anchor=(0.5, 1.2),  # distance from bottom-left
           loc='upper center',
           )
plt.show()
