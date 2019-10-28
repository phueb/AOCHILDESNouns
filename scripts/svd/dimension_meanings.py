from typing import Optional, Set

from scipy.sparse import linalg as slinalg
from sklearn.preprocessing import normalize
import numpy as np
import attr
import seaborn as sns
import matplotlib.pyplot as plt

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.pos import load_pos_words
from wordplay.svd import make_term_by_window_co_occurrence_mat, decode_singular_dimensions
from wordplay import config

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-4096'

TEST_FROM_MIDDLE = False  # put 1K docs in middle into test split

docs = load_docs(CORPUS_NAME,
                 test_from_middle=TEST_FROM_MIDDLE,
                 num_test_docs=0)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)

# /////////////////////////////////////////////////////////////////

WINDOW_SIZE = 1
NUM_DIMS = 256
NORMALIZE = True
MAX_FREQUENCY = 1000 * 100  # largest value in co-occurrence matrix
LOG_FREQUENCY = False  # take log of co-occurrence matrix element-wise

SYN_CATEGORIES: Optional[Set[str]] = {'nouns'}
SEM_CATEGORIES: Optional[Set[str]] = None

LABELS = ['noun', 'semantic']  # order matters

if SYN_CATEGORIES is None:
    SYN_CATEGORIES = {'nouns', 'verbs', 'pronouns', 'random'}

if SEM_CATEGORIES is None:
    SEM_CATEGORIES = probe_store.cats

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
start1, end1 = 0, prep.store.num_tokens
tw_mat1, xws1, yws1 = make_term_by_window_co_occurrence_mat(
    prep, start=start1, end=end1, window_size=WINDOW_SIZE, max_frequency=MAX_FREQUENCY, log=LOG_FREQUENCY)
mat = tw_mat1.T.asfptype()  # transpose so that x-words now index rows (does not affect singular values)

# normalization
if NORMALIZE:
    mat = normalize(mat, axis=1, norm='l2', copy=False)

# svd
u, s, _ = slinalg.svds(mat, k=NUM_DIMS, return_singular_vectors=True)  # s is not 2D

# /////////////////////////////////////////////////////////////////////////// decode + plot

label2dim_ids = {}
for cat2words, label in zip([syn_cat2words, sem_cat2words], LABELS):

    cat2y, dim_ids = decode_singular_dimensions(u, cat2words, prep, num_dims=NUM_DIMS)
    label2dim_ids[label] = dim_ids

    categories = cat2words.keys()

    # scatter plot
    _, ax = plt.subplots(dpi=192, figsize=(6, 6))
    ax.set_title(f'Decoding Singular Dimensions\nof term-by-window matrix\nwindow size={WINDOW_SIZE}',
                 fontsize=config.Fig.fontsize)
    # axis
    ax.set_xlabel('Singular Dimension', fontsize=config.Fig.fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False, left=False)
    ax.set_yticks([])
    ax.set_xlim(left=0, right=NUM_DIMS)
    # plot
    x = np.arange(NUM_DIMS)
    num_categories = len(categories)
    cat2color = {n: c for n, c in zip(categories, sns.color_palette("hls", num_categories))}
    for cat in categories:
        color = cat2color[cat] if cat != 'random' else 'black'
        ax.scatter(x, cat2y[cat][::-1], color=color, label=cat)

    plt.legend(frameon=True, framealpha=1.0, bbox_to_anchor=(0.5, 1.4), ncol=4, loc='lower center')
    plt.show()


# histogram has both semantic and syntactic dimensions
_, axh = plt.subplots(dpi=192, figsize=(6, 6))
axh.set_title(f'Singular Dimensions encoding syntactic categories\nwindow size={WINDOW_SIZE}',
              fontsize=config.Fig.fontsize)
axh.set_xlabel('Singular Dimension', fontsize=config.Fig.fontsize)
axh.set_ylabel('Normalized Frequency', fontsize=config.Fig.fontsize)
axh.spines['right'].set_visible(False)
axh.spines['top'].set_visible(False)
axh.tick_params(axis='both', which='both', top=False, right=False)
#
for label, dim_ids in label2dim_ids.items():
    y = np.abs(np.array(dim_ids) - NUM_DIMS)
    axh.hist(y, density=True, label=label, histtype='step', bins=None, range=[0, NUM_DIMS])
plt.legend(frameon=False, framealpha=1.0)
plt.show()