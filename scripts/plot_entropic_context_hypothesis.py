
"""
is H(X|Y) of co-occ mat reduced when more co-occurrences are in rare entropic columns?
"""
import matplotlib.pyplot as plt
from pyitlib import discrete_random_variable as drv
import numpy as np


N = 80_000
NUM_ROWS = 650
NUM_COLS = 2000
DIVISORS = 2, 3, 4, 5  # fraction of co-occurrences that are in entropic columns

x_ticks = np.arange(2, 42, 2)

# simulating rare but highly entropic contexts

divisor2xys_age1 = {divisor: [] for divisor in DIVISORS}
for divisor, xys_age1 in divisor2xys_age1.items():
    for x_tick in x_ticks:
        res = []
        for _ in range(3):
            num_entropic_observations = N // divisor
            num_remaining_observations = N - num_entropic_observations
            xs = np.hstack((np.random.randint(0, x_tick, num_entropic_observations),
                            np.random.randint(x_tick, NUM_ROWS, num_remaining_observations)))
            ys = np.random.randint(0, NUM_COLS, N)
            res_i = drv.entropy_conditional(xs, ys)
            res.append(res_i)
        divisor2xys_age1[divisor].append(np.mean(res))


# no entropic contexts - and larger matrix (simulates age group 2)

res = []
for _ in range(5):
    xs = np.random.randint(0, NUM_ROWS + 30, N)
    ys = np.random.randint(0, NUM_COLS + 600, N)
    res_i = drv.entropy_conditional(xs, ys)
    res.append(res_i)
xy_age2 = np.mean(res)
print(f'{xy_age2 :.4f}')


fig, ax = plt.subplots(1, figsize=(6, 4), dpi=300)
ax.set_ylabel('H(X|Y)', fontsize=12)
ax.set_xlabel('Number of entropy-maximizing contexts', fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks)
ax.yaxis.grid(False)
ax.set_ylim([3, 6])
# plot
ls = iter(['-', '--', '-.', ':'])
ax.axhline(y=xy_age2, color='C1', label='simulated age group 2')
for divisor, xys_age1 in divisor2xys_age1.items():
    ax.plot(x_ticks,
            xys_age1,
            color='C0',
            label=f'simulated age group 1 with fraction={divisor}',
            linestyle=next(ls))
plt.legend(frameon=False)
plt.show()