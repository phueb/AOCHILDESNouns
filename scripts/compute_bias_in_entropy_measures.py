"""
Estimates of conditional entropy, and other measures based on entropy,
are influenced not only by the number of observations, but also by the number of types.

For example, an increase in the number of columns, despite no change in the underlying distributions,
produces smaller estimates for H(X|Y), while an increase in the number of rows does not influence estimation.
Similarly, an increase in the number of rows, despite no change in the underlying distributions,
 produces larger estimates for H(Y|X), while an increase in the number of columns does not influence estimation.
"""

from pyitlib import discrete_random_variable as drv
import numpy as np


N = 144_000
# SHAPES = [(1784, 2645), (1531, 2945)]
# SHAPES = [(1784, 2645), (1531, 2645)]
SHAPES = [(1784, 2645), (1784, 2945)]

print('joint entropy')
for num_x, num_y in SHAPES:
    res = []
    for _ in range(50):
        xs = np.random.randint(0, num_x, N)
        ys = np.random.randint(0, num_y, N)
        xy = np.vstack((xs, ys))
        res_i = drv.entropy_joint(xy)
        res.append(res_i)
    print(f'{np.mean(res):.4f}')

print('conditional entropy x|y')
for num_x, num_y in SHAPES:
    res = []
    for _ in range(50):
        xs = np.random.randint(0, num_x, N)
        ys = np.random.randint(0, num_y, N)
        res_i = drv.entropy_conditional(xs, ys)
        res.append(res_i)
    print(f'{np.mean(res):.4f}')

print('conditional entropy y|x')
for num_x, num_y in SHAPES:
    res = []
    for _ in range(50):
        xs = np.random.randint(0, num_x, N)
        ys = np.random.randint(0, num_y, N)
        res_i = drv.entropy_conditional(ys, xs)
        res.append(res_i)
    print(f'{np.mean(res):.4f}')