from pyitlib import discrete_random_variable as drv
import numpy as np
from tabulate import tabulate
import pandas as pd
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.utils.extmath import randomized_svd
from sortedcontainers import SortedSet

from abstractfirst.binned import make_age_bin2data, adjust_binned_data
from abstractfirst.figs import plot_heatmap
from abstractfirst.util import to_pyitlib_format, load_words, cluster
from abstractfirst.co_occurrence import make_sparse_co_occurrence_mat
from abstractfirst.memory import set_memory_limit

set_memory_limit(0.9)

# ///////////////////////////////////////////////////////////////// parameters

CORPUS_NAME = 'childes-20201026'
AGE_STEP = 900
NUM_TOKENS_PER_BIN = 2_500_000  # 2.5M is good with AGE_STEP=900
NUM_TARGETS_IN_CO_MAT = 309_000  # or None

LEFT_ONLY = False
RIGHT_ONLY = False
REMOVE_ONES = False
SEPARATE_LEFT_AND_RIGHT = False

PLOT_HEATMAP = False

# ///////////////////////////////////////////////////////////////// separate data by age

print('Raw age bins:')
age_bin2data_ = make_age_bin2data(CORPUS_NAME, AGE_STEP)
for age_bin, data in age_bin2data_.items():
    print(f'age bin={age_bin:>4} num data={len(data):,}')

print('Adjusted age bins:')
age_bin2data = adjust_binned_data(age_bin2data_, NUM_TOKENS_PER_BIN)
for age_bin, data in sorted(age_bin2data.items(), key=lambda i: i[0]):
    print(f'age bin={age_bin:>4} num data={len(data):,}')

print(f'Number of age bins={len(age_bin2data)}')


# ///////////////////////////////////////////////////////////////// targets

types = SortedSet()
for age_bin, data in sorted(age_bin2data.items(), key=lambda i: i[0]):
    types.update(data)
targets = SortedSet([t for t in load_words('nouns-human_annotated') if t in types])


# ///////////////////////////////////////////////////////////////// init data collection

XYE = 'xye'
YXE = 'yxe'
_MI = ' mi'
NMI = 'nmi'
AMI = 'ami'
IV_ = ' iv'
JE_ = ' je'
S1MS2U = 's1-s2'
S1MS2N = 's1-s2 / s1+s2'
S1DSR_ = 's1 / sum(s)'

var_names = [XYE, YXE, _MI, NMI, AMI, IV_, JE_, S1MS2U, S1MS2N, S1DSR_]

name2col = {n: [] for n in var_names}

# ///////////////////////////////////////////////////////////////// data collection

# todo keep documents intact. feed each to spacy tagging, and then slide window over tagged doc

for age_bin, tokens in sorted(age_bin2data.items(), key=lambda i: i[0]):
    print()
    print(f'age bin={age_bin}')

    co_mat = make_sparse_co_occurrence_mat(tokens,
                                           targets,
                                           stop_n=NUM_TARGETS_IN_CO_MAT,
                                           left_only=LEFT_ONLY,
                                           right_only=RIGHT_ONLY,
                                           separate_left_and_right=SEPARATE_LEFT_AND_RIGHT,
                                           )

    # factor analysis
    print('Factor analysis...')
    u, s, v = randomized_svd(co_mat, n_components=co_mat.shape[1])
    with np.printoptions(precision=2, suppress=True):
        print(s)
    name2col[S1MS2U].append(s[0] - s[1])
    name2col[S1MS2N].append((s[0] - s[1]) / (s[0] + s[1]))
    name2col[S1DSR_].append(s[0] / np.sum(s))

    # info theory analysis  # todo figure out bits vs. nats
    print('Info theory analysis...')
    xs, ys = to_pyitlib_format(co_mat, REMOVE_ONES)
    xy = np.vstack((xs, ys))
    name2col[XYE].append(drv.entropy_conditional(xs, ys).item())
    name2col[YXE].append(drv.entropy_conditional(ys, xs).item())
    name2col[_MI].append(drv.information_mutual(xs, ys).item())
    name2col[NMI].append(drv.information_mutual_normalised(xs, ys, norm_factor='XY').item())
    name2col[AMI].append(adjusted_mutual_info_score(xs, ys, average_method="arithmetic"))
    name2col[IV_].append(drv.information_variation(xs, ys))
    name2col[JE_].append(drv.entropy_joint(xy))

    if PLOT_HEATMAP:
        co_mat_dense = co_mat.todense()
        plot_heatmap(cluster(co_mat_dense), [], [], vmax=1)  # set v-max to 1 to make all nonzero values black

# ///////////////////////////////////////////////////////////////// show data

df = pd.DataFrame(data=name2col, columns=var_names, index=age_bin2data.keys())
print(tabulate(df,
               headers=['age_onset (days)'] + var_names,
               tablefmt='simple'))
