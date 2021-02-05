"""
This script demonstrates how thr variance explained by the first singular value (s1/s)
 scales differently compared to the mutual information, when a column in a matrix is multiplied by positive integers
 AND other elements are subtracted such that the total sum of the matrix remains constant



"""
import numpy as np
import pyitlib.discrete_random_variable as drv
import matplotlib.pyplot as plt

from abstractfirst.figs import make_line_fig


mat_ = np.array([
        [20, 10, 10, 10, 10, 30],
        [10, 20, 10, 10, 10, 30],
        [10, 10, 20, 10, 10, 30],
        [10, 10, 10, 20, 10, 30],
        [10, 10, 10, 10, 20, 30],
    ])


steps = [1, 2, 3, 4, 5, 6, 7, 8, 9]
label2y = {'s1/s': [], 'I(X:Y)': []}
for step in steps:

    # subtract from diagonal & add result to last column
    mat = mat_.copy()
    mat[np.diag_indices_from(mat[:, :-1])] -= step
    mat[:, -1] += step

    xis = []
    yis = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            xis += [i] * v
            yis += [j] * v

    with np.printoptions(precision=2, linewidth=120):
        print(mat)
        print(mat.sum())

        mi = drv.information_mutual_normalised(xis, yis).round(4)

        s = np.linalg.svd(mat, compute_uv=False)
        assert np.max(s) == s[0]
        s1_norm = s[0] / np.sum(s)
        print(s.round(4))

        # collect for plotting
        label2y['s1/s'].append(s1_norm)
        label2y['I(X:Y)'].append(mi)

make_line_fig(label2y,
              'Differential scaling of s1/s vs. I(X;Y)',
              x_axis_label='step',
              y_axis_label='',
              x_ticks=steps,
              y_lims=[0, 1])
plt.show()
