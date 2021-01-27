from pyitlib import discrete_random_variable as drv
import numpy as np
from tabulate import tabulate
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.utils.extmath import randomized_svd
from sortedcontainers import SortedSet

from abstractfirst.binned import make_age_bin2data, make_age_bin2data_with_min_size
from abstractfirst.figs import plot_heatmap
from abstractfirst.util import to_pyitlib_format, load_words
from abstractfirst.co_occurrence import make_sparse_co_occurrence_mat
from abstractfirst.memory import set_memory_limit

set_memory_limit(0.9)

# ///////////////////////////////////////////////////////////////// parameters

CORPUS_NAME = 'childes-20201026'
AGE_STEP = 900
NUM_TOKENS_PER_BIN = 2_500_000  # 2.5M is good with AGE_STEP=900
NUM_TARGETS_IN_CO_MAT = 336_000  # or None

PLOT_HEATMAP = False

# ///////////////////////////////////////////////////////////////// combine docs by age

age_bin2data_ = make_age_bin2data(CORPUS_NAME, AGE_STEP)
for age_bin, data in age_bin2data_.items():
    print(f'age bin={age_bin:>4} num data={len(data):,}')

age_bin2data = make_age_bin2data_with_min_size(age_bin2data_, NUM_TOKENS_PER_BIN)
for age_bin, data in sorted(age_bin2data.items(), key=lambda i: i[0]):
    print(f'age bin={age_bin:>4} num data={len(data):,}')

print(f'Number of age bins={len(age_bin2data)}')


# ///////////////////////////////////////////////////////////////// targets

types = SortedSet()
for age_bin, data in sorted(age_bin2data.items(), key=lambda i: i[0]):
    types.update(data)
targets = SortedSet([t for t in load_words('nouns-human_annotated') if t in types])


# ///////////////////////////////////////////////////////////////// analysis

for age_bin, tokens in sorted(age_bin2data.items(), key=lambda i: i[0]):

    co_mat = make_sparse_co_occurrence_mat(tokens, targets,
                                           stop_n=NUM_TARGETS_IN_CO_MAT)

    print()
    if PLOT_HEATMAP:
        co_mat_dense = co_mat.todense()
        plot_heatmap(co_mat_dense, [], [], vmax=1)  # set v-max to 1 to make all nonzero values black

    # info theory analysis
    xs, ys = to_pyitlib_format(co_mat.todok())
    xy = np.vstack((xs, ys))
    print(f' mi={drv.information_mutual(xs, ys):2.4f}')
    print(f'nmi={drv.information_mutual_normalised(xs, ys):2.4f}')
    print(f'ami={adjusted_mutual_info_score(xs, ys, average_method="arithmetic"):2.4f}')
    print(f' iv={drv.information_variation(xs, ys):2.4f}')  # iv = je - mi
    print(f' je={drv.entropy_joint(xy):2.4f}')

    u, s, v = randomized_svd(co_mat, n_components=co_mat.shape[1])
    with np.printoptions(precision=2, suppress=True):
        print(s)
        print(f's1-s2={s[0]-s[1]}')