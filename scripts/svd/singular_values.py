"""
Research questions:
1. Are singular values higher for first or second half of input?
2. Are those dimensions with higher singular values NOUN-encoding dimensions?
"""

import matplotlib.pyplot as plt
from scipy.sparse import linalg as slinalg
from sklearn.preprocessing import normalize
import attr
import numpy as np

from preppy.legacy import TrainPrep

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.svd import make_context_by_term_matrix
from wordplay.svd import decode_singular_dimensions
from wordplay.pos import load_pos_words
from wordplay import config

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'

SHUFFLE_DOCS = False
NUM_MID_TEST_DOCS = 0

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=NUM_MID_TEST_DOCS,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams(num_types=None)
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

CONTEXT_SIZE = 2
NUM_DIMS = 32
NORMALIZE = False  # this makes all the difference - this means that the scales of variables are different and matter


NOM_ALPHA = 0.01  # TODO test

OFFSET = prep.midpoint
LABELS = [f'first {OFFSET:,} tokens', f'last {OFFSET:,} tokens']

# ///////////////////////////////////////////////////////////////////// categories

# make syntactic categories for probing
cat2words = {}
for cat in ['nouns', 'verbs']:
    category_words = load_pos_words(f'{CORPUS_NAME}-{cat}')
    cat2words[cat] = category_words
    print(f'Loaded {len(category_words)} words in category {cat}')
    assert len(category_words) > 0

# /////////////////////////////////////////////////////////////////////////// SVD

# make term_by_window_co_occurrence_mats
start1, end1 = 0, prep.midpoint
start2, end2 = prep.midpoint, prep.store.num_tokens
tw_mat1, xws1, yws1 = make_context_by_term_matrix(
    prep.store.tokens, start=start1, end=end1, context_size=CONTEXT_SIZE)
tw_mat2, xws2, yws2 = make_context_by_term_matrix(
    prep.store.tokens, start=start2, end=end2, context_size=CONTEXT_SIZE)


# collect singular values
label2cat2dim_ids = {}
label2s = {}
for mat, label, x_words in zip([tw_mat1.T.asfptype(), tw_mat2.T.asfptype()],
                               LABELS,
                               [xws1, xws2]):

    if NORMALIZE:
        mat = normalize(mat, axis=1, norm='l2', copy=False)

    # SVD
    u, s, _ = slinalg.svds(mat, k=NUM_DIMS, return_singular_vectors=True)
    print('sum of singular values={:,}'.format(s.sum()))
    print('var of singular values={:,}'.format(s.var()))

    # collect singular values
    label2s[label] = s

    # let noun and verbs compete for dimensions to be sure that a dimension encodes nouns
    cat2dim_ids = decode_singular_dimensions(u, cat2words, x_words,
                                             num_dims=NUM_DIMS,
                                             nominal_alpha=NOM_ALPHA,
                                             plot_loadings=False)

    label2cat2dim_ids[label] = cat2dim_ids

# get noun dims to label figure
noun_dims1 = label2cat2dim_ids[LABELS[0]]['nouns']
noun_dims2 = label2cat2dim_ids[LABELS[1]]['nouns']
s1 = label2s[LABELS[0]]
s2 = label2s[LABELS[1]]
s1_noun_dims = [s1[i] if not np.isnan(i) else np.nan for i in noun_dims1]
s2_noun_dims = [s2[i] if not np.isnan(i) else np.nan for i in noun_dims2]

# figure
fig, ax = plt.subplots(1, figsize=(5, 5), dpi=None)
plt.title(f'SVD of AO-CHILDES partitions\nwindow size={CONTEXT_SIZE}', fontsize=config.Fig.fontsize)
ax.set_ylabel('Singular value', fontsize=config.Fig.fontsize)
ax.set_xlabel('Singular Dimension', fontsize=config.Fig.fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
ax.plot(s1[::-1], label=LABELS[0], linewidth=2, color='C0')
ax.plot(s2[::-1], label=LABELS[1], linewidth=2, color='C1')
# label noun-dims
x = np.arange(NUM_DIMS)

ax.scatter(x, s1_noun_dims[::-1], label='NOUN dimension', color='C0', zorder=3)
ax.scatter(x, s2_noun_dims[::-1], label='NOUN dimension', color='C1', zorder=3)
ax.legend(loc='upper right', frameon=False, fontsize=config.Fig.fontsize)
plt.tight_layout()
plt.show()