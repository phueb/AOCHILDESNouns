import numpy as np
import pandas as pd
import spacy
from typing import List
from spacy.tokens import Token, Doc
from pyitlib import discrete_random_variable as drv
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sortedcontainers import SortedSet
from tabulate import tabulate

from abstractfirst.binned import make_age_bin2data, adjust_binned_data
from abstractfirst.co_occurrence import make_sparse_co_occurrence_mat
from abstractfirst.figs import plot_heatmap
from abstractfirst.memory import set_memory_limit
from abstractfirst.util import to_pyitlib_format, load_words, cluster

set_memory_limit(0.9)

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# ///////////////////////////////////////////////////////////////// parameters

CORPUS_NAME = 'childes-20201026'
AGE_STEP = 900
NUM_TOKENS_PER_BIN = 2_500_000  # 2.5M is good with AGE_STEP=900
NUM_TARGETS_IN_CO_MAT = None  # or None
FILTER_TARGETS = False

LEFT_ONLY = False
RIGHT_ONLY = False
REMOVE_ONES = False
SEPARATE_LEFT_AND_RIGHT = True

PLOT_HEATMAP = False

# ///////////////////////////////////////////////////////////////// separate data by age

print('Raw age bins:')
age_bin2text_unadjusted = make_age_bin2data(CORPUS_NAME, AGE_STEP)
for age_bin, txt in age_bin2text_unadjusted.items():
    print(f'age bin={age_bin:>4} num tokens={len(txt.split()):,}')

print('Adjusted age bins:')
age_bin2text = adjust_binned_data(age_bin2text_unadjusted, NUM_TOKENS_PER_BIN)
for age_bin, txt in sorted(age_bin2text.items(), key=lambda i: i[0]):
    num_tokens_adjusted = len(txt.split())
    print(f'age bin={age_bin:>4} num tokens={num_tokens_adjusted :,}')
    assert num_tokens_adjusted >= NUM_TOKENS_PER_BIN

print(f'Number of age bins={len(age_bin2text)}')

# ///////////////////////////////////////////////////////////////// targets

targets_allowed = SortedSet(load_words('nouns-human_annotated'))

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

for age_bin, text in sorted(age_bin2text.items(), key=lambda i: i[0]):
    print()
    print(f'age bin={age_bin}')

    print('Tagging...')
    nlp.max_length = len(text)
    doc: Doc = nlp(text)  # todo use pipe() and input docs with doc boundaries intact
    print(f'Found {len(doc):,} tokens in text')

    co_mat: sparse.coo_matrix = make_sparse_co_occurrence_mat(doc,
                                                              targets_allowed,
                                                              stop_n=NUM_TARGETS_IN_CO_MAT,
                                                              left_only=LEFT_ONLY,
                                                              right_only=RIGHT_ONLY,
                                                              separate_left_and_right=SEPARATE_LEFT_AND_RIGHT,
                                                              )

    # factor analysis
    print('Factor analysis...')
    co_mat_csc: sparse.csc_matrix = co_mat.tocsc()
    co_mat_csc = co_mat.asfptype()
    s_low_to_high = svds(co_mat_csc, k=min(co_mat_csc.shape) - 1, return_singular_vectors=False)
    s = s_low_to_high[::-1]
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

df = pd.DataFrame(data=name2col, columns=var_names, index=age_bin2text.keys())
print(tabulate(df,
               headers=['age_onset (days)'] + var_names,
               tablefmt='simple'))
