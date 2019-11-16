"""
Research questions:
1. Which singualr value dimensions of term-window matrix encode which semantic categories?
2. How does the first vs. second half of the input vary in which semantic categories are encoded?
3. Is overall more variance accounted for by semantic-encoding dimensions in the first or second half of input?

"""


import matplotlib.pyplot as plt
from scipy.sparse import linalg as slinalg
from sklearn.preprocessing import normalize
import numpy as np

import attr

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay import config
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.pos import load_pos_words
from wordplay.docs import load_docs
from wordplay.representation import make_context_by_term_matrix
from wordplay.svd import decode_singular_dimensions
from wordplay.svd import plot_category_encoding_dimensions
from wordplay import config

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-all'

SHUFFLE_DOCS = False
NUM_MID_TEST_DOCS = 0

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=NUM_MID_TEST_DOCS,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

CONTEXT_SIZE = 1
NUM_DIMS = 256
NORMALIZE = False  # this makes all the difference - this means that the scales of variables are different and matter


NOM_ALPHA = 0.01

OFFSET = prep.midpoint
LABELS = [f'first {OFFSET:,} tokens', f'last {OFFSET:,} tokens']

SCATTER_PLOT = True

# ///////////////////////////////////////////////////////////////////// categories

nouns = load_pos_words(f'{CORPUS_NAME}-nouns')

# make semantic categories for probing
probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)
cat2words = {}
for cat in probe_store.cats:
    category_words = probe_store.cat2probes[cat]
    cat2words[cat] = category_words
    print(f'Loaded {len(category_words)} words in category {cat}')
    assert len(category_words) > 0

categories = cat2words.keys()
num_categories = len(categories)

# ////////////////////////////////////////////////////////////////////// TW matrix

# make term_by_window_co_occurrence_mats
start1, end1 = 0, OFFSET
start2, end2 = prep.store.num_tokens - OFFSET, prep.store.num_tokens
tw_mat1, xws1, yws1 = make_context_by_term_matrix(prep.store.tokens, start=start1, end=end1, context_size=CONTEXT_SIZE)
tw_mat2, xws2, yws2 = make_context_by_term_matrix(prep.store.tokens, start=start2, end=end2, context_size=CONTEXT_SIZE)


# ////////////////////////////////////////////////////////////////////// svd

label2cat2dim_ids = {}
label2s = {}
for mat, label, x_words in zip([tw_mat1.T.asfptype(), tw_mat2.T.asfptype()],
                               LABELS,
                               [xws1, xws2]):

    if NORMALIZE:
        print('Normalizing...')
        mat = normalize(mat, axis=1, norm='l2', copy=False)

    # SVD
    print('Fitting SVD ...')
    u, s, _ = slinalg.svds(mat, k=NUM_DIMS, return_singular_vectors=True)  # s is not 2D
    label2s[label] = s

    # /////////////////////////////////////////////////////////////////////////// decode + plot

    cat2dim_ids = decode_singular_dimensions(u, cat2words, x_words,
                                             num_dims=NUM_DIMS,
                                             nominal_alpha=NOM_ALPHA,
                                             plot_loadings=False,
                                             control_words=nouns)

    dim_ids = []
    for cat, idx in cat2dim_ids.items():
        dim_ids_without_nans = [i for i in idx if not np.isnan(i)]
        dim_ids += dim_ids_without_nans
    assert len(dim_ids) == len(set(dim_ids))  # do not accept duplicates

    label2cat2dim_ids[label] = cat2dim_ids

    print('Dimensions identified as encoding semantic-category-membership:')
    print(dim_ids)

    if SCATTER_PLOT:
        title = f'Decoding Singular Dimensions\nof {label} term-by-window matrix\nwindow size={CONTEXT_SIZE}'
        plot_category_encoding_dimensions(cat2dim_ids, NUM_DIMS, title)


# comparing singular values - does syntactic or semantic category account for more?
_, ax = plt.subplots(dpi=config.Fig.dpi, figsize=config.Fig.fig_size)
ax.set_title(f'Variance explained by {PROBES_NAME}-encoding dimensions\nwindow size={CONTEXT_SIZE}', fontsize=config.Fig.ax_fontsize)
ax.set_ylabel('Normalized Singular Value', fontsize=config.Fig.ax_fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.set_xticklabels(LABELS)
plt.grid(True, which='both', axis='y', alpha=0.2)
plt.yscale('log')
# y-lims
if NORMALIZE:
    ylims = [0, 5]
elif LOG_FREQUENCY:
    ylims = [10e-6, 10e-2]
else:
    ylims = [0, 20]
ax.set_ylim(ylims)
#
s1 = label2s[LABELS[0]]
s2 = label2s[LABELS[1]]
s1_all_dims = label2s[LABELS[0]]
s2_all_dims = label2s[LABELS[1]]
assert np.count_nonzero(s1) == len(s1)
assert np.count_nonzero(s2) == len(s2)
# dimensions
dims1_nans = np.unique([label2cat2dim_ids[LABELS[0]][cat] for cat in categories])
dims2_nans = np.unique([label2cat2dim_ids[LABELS[1]][cat] for cat in categories])
dims1 = [int(i) for i in dims1_nans if not np.isnan(i)]
dims2 = [int(i) for i in dims2_nans if not np.isnan(i)]
# singular values
s1_sem_dims = np.array([s1[i] for i in dims1])  # sing values for part 1 sem dimensions
s2_sem_dims = np.array([s2[i] for i in dims2])  # sing values for part 2 sem dimensions
total_var1 = s1_all_dims.sum()
total_var2 = s2_all_dims.sum()
# plot
y1 = s1_sem_dims[::-1] / total_var1
y2 = s2_sem_dims[::-1] / total_var2
ax.boxplot([y1, y2], zorder=3)
ax.axhline(y=np.mean(y1), label=f'part 1 mean={np.mean(y1):.4f} n={len(y1)}', color='blue')
ax.axhline(y=np.mean(y2), label=f'part 2 mean={np.mean(y2):.4f} n={len(y2)}', color='red')
plt.legend(loc='lower left', framealpha=1.0)
plt.show()


# proportion of total variance accounted for by dimensions encoding semantics
print(f'{s1_sem_dims.sum():>12,.1f} {total_var1 :>12,.1f} {s1_sem_dims.sum() / total_var1 :>6,.3f}')
print(f'{s2_sem_dims.sum():>12,.1f} {total_var2 :>12,.1f} {s2_sem_dims.sum() / total_var2 :>6,.3f}')

# proportion of variance by category
print(f'category     prop1  prop2')
for cat in categories:
    cat_dim_ids1 = [i for i in label2cat2dim_ids[LABELS[0]][cat] if not np.isnan(i)]
    cat_dim_ids2 = [i for i in label2cat2dim_ids[LABELS[1]][cat] if not np.isnan(i)]
    cat_var1 = np.sum([label2s[LABELS[0]][i] for i in cat_dim_ids1])
    cat_var2 = np.sum([label2s[LABELS[1]][i] for i in cat_dim_ids2])
    prop1 = cat_var1 / total_var1
    prop2 = cat_var2 / total_var2
    print(f'{cat:<12}{prop1 :>6,.3f} {prop2 :>6,.3f} | {prop1 / prop2: .2f}')