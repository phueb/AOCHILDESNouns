import matplotlib.pyplot as plt
from scipy.sparse import linalg as slinalg
from sklearn.preprocessing import normalize
import numpy as np
import seaborn as sns

import attr

from preppy.legacy import TrainPrep

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.svd import make_term_by_window_co_occurrence_mat, decode_singular_dimensions
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

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

WINDOW_SIZE = 2
NUM_DIMS = 256
NORMALIZE = False  # this makes all the difference - this means that the scales of variables are different and matter
MAX_FREQUENCY = 100 * 1000  # largest value in co-occurrence matrix
LOG_FREQUENCY = True  # take log of co-occurrence matrix element-wise

NOM_ALPHA = 0.05

OFFSET = prep.midpoint
LABELS = [f'first {OFFSET:,} tokens',
          f'last {OFFSET:,} tokens']

SCATTER_PLOT = False

# make term_by_window_co_occurrence_mats
start1, end1 = 0, OFFSET
start2, end2 = prep.store.num_tokens - OFFSET, prep.store.num_tokens
tw_mat1, xws1, yws1 = make_term_by_window_co_occurrence_mat(
    prep, start=start1, end=end1, window_size=WINDOW_SIZE, max_frequency=MAX_FREQUENCY, log=LOG_FREQUENCY)
tw_mat2, xws2, yws2 = make_term_by_window_co_occurrence_mat(
    prep, start=start2, end=end2, window_size=WINDOW_SIZE, max_frequency=MAX_FREQUENCY, log=LOG_FREQUENCY)


# ////////////////////////////////////////////////////////////////////// svd

label2dim_ids = {}
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

    nouns = load_pos_words(f'{CORPUS_NAME}-nouns')
    rands = load_pos_words(f'{CORPUS_NAME}-random')
    print(f'Loaded {len(nouns)} nouns')
    cat2words = {'noun': nouns, 'random': rands}
    categories = cat2words.keys()

    cat2y, dim_ids = decode_singular_dimensions(u, cat2words, x_words,
                                                num_dims=NUM_DIMS, nominal_alpha=NOM_ALPHA, plot_loadings=False)
    label2dim_ids[label] = dim_ids

    print('Dimensions identified as encoding noun-membership:')
    print(dim_ids)

    if SCATTER_PLOT:

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


# comparing singular values - does syntactic or semantic category account for more?
_, ax = plt.subplots(dpi=192, figsize=(6, 6))
ax.set_title(f'Variance explained by NOUN-encoding dimensions\nwindow size={WINDOW_SIZE}', fontsize=config.Fig.fontsize)
ax.set_ylabel('Normalized Singular Value', fontsize=config.Fig.fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.set_xticklabels(LABELS)
ax.yaxis.grid(True, alpha=0.5)
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


# plot
s1 = label2s[LABELS[0]]
s2 = label2s[LABELS[1]]
s1_all_dims = label2s[LABELS[0]]
s2_all_dims = label2s[LABELS[1]]
assert np.count_nonzero(s1) == len(s1)
assert np.count_nonzero(s2) == len(s2)
s1_noun_dims = np.array([s1[::-1][i] for i in label2dim_ids[LABELS[0]]])  # singular values for part 1 noun dimensions
s2_noun_dims = np.array([s2[::-1][i] for i in label2dim_ids[LABELS[1]]])  # singular values for part 2 noun dimensions
ax.boxplot([s1_noun_dims / s1_all_dims.sum(), s2_noun_dims / s2_all_dims.sum()], zorder=3)
plt.show()


# proportion of total variance
print(s1_noun_dims.sum(), s1_all_dims.sum(), s1_noun_dims.sum() / s1_all_dims.sum())
print(s2_noun_dims.sum(), s2_all_dims.sum(), s2_noun_dims.sum() / s2_all_dims.sum())