from pyitlib import discrete_random_variable as drv
import numpy as np


N = 144_000
SHAPES = [(1784, 2645), (1531, 2945)]
# SHAPES = [(1784, 2645), (1531, 2645)]
# SHAPES = [(1784, 2645), (1784, 2945)]

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

# print('conditional entropy y|x')
# for num_x, num_y in SHAPES:
#     res = []
#     for _ in range(50):
#         xs = np.random.randint(0, num_x, N)
#         ys = np.random.randint(0, num_y, N)
#         res_i = drv.entropy_conditional(ys, xs)
#         res.append(res_i)
#     print(f'{np.mean(res):.4f}')