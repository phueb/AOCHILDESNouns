import numpy as np
import pyitlib.discrete_random_variable as drv
import matplotlib.pyplot as plt

from abstractfirst.figs import make_line_fig


mat_ = np.array([
        [1, 1, 1, 5, 1],
        [1, 1, 1, 5, 1],
        [1, 1, 1, 5, 1],
        [1, 5, 1, 1, 1],
    ]) * 10


multipliers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
label2y = {'s1/s': [], 'I(X:Y)': []}
for multiplier in multipliers:

    mat = mat_.copy()
    mat[:, 3] *= multiplier

    xis = []
    yis = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            xis += [i] * v
            yis += [j] * v

    with np.printoptions(precision=2, linewidth=120):
        print(mat)
        print(np.linalg.matrix_rank(mat))

        mi = drv.information_mutual_normalised(xis, yis).round(4)

        s = np.linalg.svd(mat, compute_uv=False)
        assert np.max(s) == s[0]
        s1_norm = s[0] / np.sum(s)

        # collect for plotting
        label2y['s1/s'].append(s1_norm)
        label2y['I(X:Y)'].append(mi)

make_line_fig(label2y,
              'Differential scaling of s1/s vs. I(X;Y)',
              x_axis_label='column multiplier',
              y_axis_label='',
              x_ticks=multipliers,
              y_lims=[0, 1])
plt.show()
