"""
Research questions:
1. Which singular dimensions of term-window matrix encode semantic or syntactic categories?
2. Do syntactic or semantic category-encoding dimensions account for more total variance?

"""


from typing import Optional, Set

from scipy.sparse import linalg as slinalg
from sklearn.preprocessing import normalize
import numpy as np
import attr

import matplotlib.pyplot as plt

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.pos import load_pos_words
from wordplay.representation import make_context_by_term_matrix
from wordplay.svd import decode_singular_dimensions
from wordplay.svd import plot_category_encoding_dimensions
from wordplay import config

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-all'

SHUFFLE_DOCS = False
NUM_MID_TEST_DOCS = 100

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=NUM_MID_TEST_DOCS,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)

# /////////////////////////////////////////////////////////////////

CONTEXT_SIZE = 2
NUM_DIMS = 256
NORMALIZE = False


SYN_CATEGORIES: Optional[Set[str]] = {'nouns', 'verbs'}
SEM_CATEGORIES: Optional[Set[str]] = None

LABELS = ['noun', 'semantic']  # order matters

if SYN_CATEGORIES is None:
    SYN_CATEGORIES = {'nouns', 'verbs', 'pronouns', 'random'}

if SEM_CATEGORIES is None:
    SEM_CATEGORIES = probe_store.cats

SCATTER_PLOT = True

# /////////////////////////////////////////////////////////////////////////// categories


# make syntactic categories for probing
syn_cat2words = {}
for cat in SYN_CATEGORIES:
    category_words = load_pos_words(f'{CORPUS_NAME}-{cat}')
    syn_cat2words[cat] = category_words
    print(f'Loaded {len(category_words)} words in category {cat}')
    assert len(category_words) > 0

# make semantic categories for probing
probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)
sem_cat2words = {}
for cat in SEM_CATEGORIES:
    category_words = probe_store.cat2probes[cat]
    sem_cat2words[cat] = category_words
    print(f'Loaded {len(category_words)} words in category {cat}')
    assert len(category_words) > 0

# add random category
SEM_CATEGORIES.add('random')
sem_cat2words['random'] = np.random.choice(probe_store.types, size=30, replace=False)

# /////////////////////////////////////////////////////////////////////////// SVD

# make term_by_window_co_occurrence_mats
tw_mat1, xws1, yws1 = make_context_by_term_matrix(prep.store.tokens,
                                                  context_size=CONTEXT_SIZE)
mat = tw_mat1.T.asfptype()  # transpose so that x-words now index rows (does not affect singular values)

# normalization
if NORMALIZE:
    mat = normalize(mat, axis=1, norm='l2', copy=False)

# svd
u, s, _ = slinalg.svds(mat, k=NUM_DIMS, return_singular_vectors=True)  # s is not 2D

# /////////////////////////////////////////////////////////////////////////// decode + plot

label2dim_ids = {}
for cat2words, label in zip([syn_cat2words, sem_cat2words], LABELS):

    cat2dim_ids = decode_singular_dimensions(u, cat2words, prep.store.types, num_dims=NUM_DIMS)

    dim_ids = []
    for cat, idx in cat2dim_ids.items():
        dim_ids_without_nans = [i for i in idx if not np.isnan(i)]
        dim_ids += dim_ids_without_nans
    assert len(dim_ids) == len(set(dim_ids))  # do not accept duplicates

    label2dim_ids[label] = dim_ids

    print(f'Dimensions identified as encoding {label}-membership:')
    print(dim_ids)

    if SCATTER_PLOT:
        title = f'Decoding Singular Dimensions\nof {label} term-by-window matrix\nwindow size={CONTEXT_SIZE}'
        plot_category_encoding_dimensions(cat2dim_ids, NUM_DIMS, title)

# comparing singular values - does syntactic or semantic category account for more?
_, ax = plt.subplots(dpi=192, figsize=(6, 6))
ax.set_title(f'Variance explained\nnouns vs. semantics\nwindow size={CONTEXT_SIZE}', fontsize=config.Fig.fontsize)
ax.set_xlabel('Category', fontsize=config.Fig.fontsize)
ax.set_ylabel('Singular Value', fontsize=config.Fig.fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.set_xticklabels(LABELS)
plt.grid(True, which='both', axis='y', alpha=0.2)
plt.yscale('log')
#
y1 = [s[::-1][i] for i in label2dim_ids[LABELS[0]]]  # singular values for syntactic categories
y2 = [s[::-1][i] for i in label2dim_ids[LABELS[1]]]  # singular values for semantic categories
ax.boxplot([y1, y2], labels=LABELS)
ax.axhline(y=np.mean(y1), label=f'noun n={len(y1)}', color='blue')
ax.axhline(y=np.mean(y2), label=f'semantics n={len(y2)}', color='red')
plt.legend()
plt.show()