
"""
there is never bias in these simulations because i never specify a non-random joint distribtuin.
all dat is randomly generated from marginal distributions.
this means predictions are for de-biased estimates of AO-CHILDES data always
"""
import matplotlib.pyplot as plt
from pyitlib import discrete_random_variable as drv
import numpy as np


N = 80_000
NUM_ROWS = 600
NUM_COLS = 2000
FRACTIONS = [2, 3, 4]

RESPECT_SHAPE_DIFF = True

x_ticks = np.arange(2, 80, 4)

# simulating rare but highly entropic contexts

p_x = np.array([1 / (i + 1) for i in range(NUM_ROWS)])
p_x = p_x / p_x.sum()

p_y = np.array([1 / (i + 1) for i in range(NUM_COLS)])
p_y = p_y / p_y.sum()

vocab_x = np.arange(NUM_ROWS)
vocab_y = np.arange(NUM_COLS)

fraction2xys_age1 = {f: [] for f in FRACTIONS}
for fraction in fraction2xys_age1:
    for x_tick in x_ticks:
        num_entropic_observations = N // fraction
        num_remaining_observations = N - num_entropic_observations

        xs = [np.random.choice(vocab_x, size=N, p=p_x)
              for _ in range(2)]
        ys = [np.hstack((np.random.choice(vocab_y[:x_tick],
                                          size=num_entropic_observations,
                                          p=None),  # no probabilities because these contexts are entropic
                         np.random.choice(vocab_y[x_tick:],
                                          size=num_remaining_observations,
                                          p=p_y[x_tick:] / p_y[x_tick:].sum())))
              for _ in range(2)]

        raw = np.mean([drv.entropy_conditional(x, y)
                       for x, y in zip(xs, ys)])  # raw is never biased

        fraction2xys_age1[fraction].append(raw)


# no entropic contexts - and larger shape (simulates age group 2)

if RESPECT_SHAPE_DIFF:
    num_rows_age2 = NUM_ROWS + 30
    num_cols_age2 = NUM_ROWS + 600
else:
    num_rows_age2 = NUM_ROWS + 0
    num_cols_age2 = NUM_ROWS + 0

xs = [np.random.choice(vocab_x, size=N, p=p_x)
      for _ in range(2)]
ys = [np.random.choice(vocab_y, size=N, p=p_y)
      for _ in range(2)]
res_age2_raw = np.mean([drv.entropy_conditional(x, y)
                        for x, y in zip(xs, ys)])
bias = np.mean([drv.entropy_conditional(np.random.permutation(x),
                                        np.random.permutation(y))
                for x, y in zip(xs, ys)])

res_age2_bias = bias

print(res_age2_raw.round(2), res_age2_bias.round(2))


fig, ax = plt.subplots(1, figsize=(6, 4), dpi=300)
ax.set_ylabel('H(X|Y)', fontsize=12)
ax.set_xlabel('Number of entropy-maximizing contexts', fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks)
ax.yaxis.grid(False)
# ax.set_ylim([-1, 1])
# ax.set_title(f'RESPECT_SHAPE_DIFF={RESPECT_SHAPE_DIFF}')
# plot
ls = iter(['-', '--', '-.', ':'])
ax.axhline(y=res_age2_raw, linestyle='-', color='C1', label='simulated age group 2')
for fraction, res_age1 in fraction2xys_age1.items():

    ax.plot(x_ticks,
            res_age1,
            color='C0',
            label=f'simulated age group 1 with fraction={fraction}',
            linestyle=next(ls))
plt.legend(frameon=False)
plt.show()