
"""
there is never bias in these simulations because i never specify a non-random joint distribution.
all dat is randomly generated from marginal distributions.
this means predictions are for de-biased estimates of AO-CHILDES data always
"""
import matplotlib.pyplot as plt
from pyitlib import discrete_random_variable as drv
import numpy as np


N = 80_000
NUM_ROWS_AGE1 = 600
NUM_COLS_AGE1 = 2000  # age group 1
FRACTIONS = [4, 8, 16]

x_ticks = np.arange(2, 2000, 50)


def compute_ce(num_add_rows: int = 0,
               num_add_cols: int = 0,
               ):

    vx = np.arange(NUM_ROWS_AGE1 + num_add_rows)
    vy = np.arange(NUM_COLS_AGE1 + num_add_cols)
    px = np.array([1 / (i + 1) for i in range(NUM_ROWS_AGE1 + num_add_rows)])
    px = px / px.sum()
    py = np.array([1 / (i + 1) for i in range(NUM_COLS_AGE1 + num_add_cols)])
    py = py / py.sum()

    # get non-entropic observations
    x_ = np.random.choice(vx, size=N, p=px)
    y_ = np.random.choice(vy, size=N, p=py)

    # compute
    xy_ = np.vstack((x_, y_))
    res = drv.entropy_conditional(x_, y_) / drv.entropy_joint(xy_)

    return res


# simulating rare but highly entropic contexts

vocab_x = np.arange(NUM_ROWS_AGE1)
vocab_y = np.arange(NUM_COLS_AGE1)
p_x = np.array([1 / (i + 1) for i in range(NUM_ROWS_AGE1)])
p_x = p_x / p_x.sum()
p_y = np.array([1 / (i + 1) for i in range(NUM_COLS_AGE1)])
p_y = p_y / p_y.sum()


fraction2xy_ces_age1 = {f: [] for f in FRACTIONS}
for fraction in fraction2xy_ces_age1:
    for x_tick in x_ticks:
        num_entropic_observations = N // fraction
        num_remaining_observations = N - num_entropic_observations

        xy_ces = []
        for _ in range(10):
            # get non-entropic observations corresponding to age group 2 shape
            x = np.random.choice(vocab_x, size=num_remaining_observations, p=p_x).tolist()
            y = np.random.choice(vocab_y, size=num_remaining_observations, p=p_y).tolist()

            # get entropic observations
            x += np.random.choice(vocab_x, size=num_entropic_observations, p=p_x).tolist()
            y += np.random.choice(vocab_y[:x_tick], size=num_entropic_observations, p=None).tolist()

            # compute
            xy = np.vstack((x, y))
            xy_ces_i = drv.entropy_conditional(x, y) / drv.entropy_joint(xy)

            # collect
            xy_ces.append(xy_ces_i)

        fraction2xy_ces_age1[fraction].append(np.mean(xy_ces))

        print(f'x_tick={x_tick:>6} fraction={fraction:>6} ce={np.mean(xy_ces)}')


xy_ce_age1 = np.mean([compute_ce(num_add_rows=0, num_add_cols=0) for _ in range(10)])
xy_ce_age2 = np.mean([compute_ce(num_add_rows=+30, num_add_cols=+600) for _ in range(10)])

print(xy_ce_age1)
print(xy_ce_age2)


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
# plot
ls = iter(['--', '-.', ':'])
ax.axhline(y=xy_ce_age2, linestyle='-', color='C1', label='simulated age group 2')
ax.axhline(y=xy_ce_age1, linestyle='-', color='C0', label='simulated age group 1 with fraction=inf')
for fraction, xy_ce_age1 in fraction2xy_ces_age1.items():
    ax.plot(x_ticks,
            xy_ce_age1,
            color='C0',
            label=f'simulated age group 1 with fraction={fraction}',
            linestyle=next(ls))
plt.legend(frameon=False)
plt.show()