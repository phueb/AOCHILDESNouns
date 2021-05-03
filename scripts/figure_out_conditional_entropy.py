from pyitlib import discrete_random_variable as drv
import numpy as np


N = 144_000
SHAPES = [(1784, 2645), (1531, 2945)]



# simulating rare but highly entropic contexts

res = []
for _ in range(5):
    xs = np.hstack((np.random.randint(0, 4, N // 2), np.random.randint(4, 1784, N // 2)))
    # xs = np.hstack((np.random.randint(0, 1784, N // 2), np.random.randint(0, 1784, N // 2)))
    ys = np.random.randint(0, 2645, N)
    res_i = drv.entropy_conditional(xs, ys)
    res.append(res_i)
print(f'{np.mean(res):.4f}')

# everything random

res = []
for _ in range(5):
    xs = np.random.randint(0, 1784, N)
    ys = np.random.randint(0, 2645, N)
    res_i = drv.entropy_conditional(xs, ys)
    res.append(res_i)
print(f'{np.mean(res):.4f}')
