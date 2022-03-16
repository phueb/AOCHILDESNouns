
"""
computed conditional entropy H(X|Y) of simulated co-occurrence matrices for age group 1 and 2
while accounting for differences in their shapes.

there are more columns in age group 2, which reduces estimate of H(X|Y) due to chance alone.
"""
import matplotlib.pyplot as plt
from pyitlib import discrete_random_variable as drv
import numpy as np


N = 80_000  # number of total co-occurrences
CO_MAT_SHAPE_1 = (600, 2000)
CO_MAT_SHAPE_2 = (630, 2600)
PROPORTIONS = [0.1, 0.5]  # proportion of contexts that are entropy-maximizing
NUM_REPEAT = 2

x_ticks = np.arange(2, 2000, 50)  # 2, 2000, 50


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
    compute normalized conditional entropy, given a vocab, and the proportion of contexts that are entropy-maximizing

    normalization does not really have a purpose here - it does not de-bias due to shape differences

    """

    num_entropic_observations = int(N * prop_)
    num_remaining_observations = N - num_entropic_observations

    # get non-entropic observations
    x = np.random.choice(vocab.x, size=num_remaining_observations, p=vocab.p_x).tolist()
    y = np.random.choice(vocab.y, size=num_remaining_observations, p=vocab.p_y).tolist()

    # get entropic observations
    x += np.random.choice(vocab.x, size=num_entropic_observations, p=vocab.p_x).tolist()
    y += np.random.choice(vocab.y[:x_tick], size=num_entropic_observations, p=None).tolist()

    # compute
    xy = np.vstack((x, y))
    res = drv.entropy_conditional(x, y) / drv.entropy_joint(xy)

    return res


vocab1 = Vocab(*CO_MAT_SHAPE_1)
vocab2 = Vocab(*CO_MAT_SHAPE_2)

prop2xy_ces_age1 = {f: [] for f in PROPORTIONS}
prop2xy_ces_age2 = {f: [] for f in PROPORTIONS}
for prop in prop2xy_ces_age1:

    for x_tick in x_ticks:

        # compute multiple times
        xy_ces1 = []
        xy_ces2 = []
        for _ in range(NUM_REPEAT):
            xy_ces1_i = compute_nce(prop, vocab1)
            xy_ces2_i = compute_nce(prop, vocab2)
            # collect
            xy_ces1.append(xy_ces1_i)
            xy_ces2.append(xy_ces2_i)

        prop2xy_ces_age1[prop].append(np.mean(xy_ces1))
        prop2xy_ces_age2[prop].append(np.mean(xy_ces2))

        print(f'x_tick={x_tick:>6} prop={prop:>6} age group1 ce={np.mean(xy_ces1)}')
        print(f'x_tick={x_tick:>6} prop={prop:>6} age group2 ce={np.mean(xy_ces2)}')
        print()


# fig
fig, ax = plt.subplots(1, figsize=(6, 4), dpi=300)
ax.set_ylabel('Normalized H(X|Y)', fontsize=12)
ax.set_xlabel('Number of entropy-maximizing contexts', fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
x_tick_labels = np.arange(0, x_ticks[-1], 200)
ax.set_xticks(x_tick_labels)
ax.set_xticklabels(x_tick_labels)
ax.yaxis.grid(False)
# plot age group1
ls = iter(['--', '-.', ':'])
for prop, xy_ce_age1 in prop2xy_ces_age1.items():
    ax.plot(x_ticks,
            xy_ce_age1,
            color='C0',
            label=f'simulated age group 1 with proportion={prop}',
            linestyle=next(ls))
# plot age group1
ls = iter(['--', '-.', ':'])
for prop, xy_ce_age2 in prop2xy_ces_age2.items():
    ax.plot(x_ticks,
            xy_ce_age2,
            color='C1',
            label=f'simulated age group 2 with proportion={prop}',
            linestyle=next(ls))

plt.legend(frameon=False)
plt.show()
