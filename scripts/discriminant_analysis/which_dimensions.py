"""
Research questions:
1. Which singular dimensions of input result in good semantic or syntactic category discrimination?
"""

import attr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.sparse.linalg as sparse_linalg

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.svd import make_term_by_context_co_occurrence_mat

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'syn-nva'  # change this to analyze semantic or syntactic categories

SHUFFLE_DOCS = False
NUM_MID_TEST_DOCS = 0

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=NUM_MID_TEST_DOCS,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

CONTEXT_SIZE = 4
NUM_DIMS = 512
NORMALIZE = False  # this makes all the difference - this means that the scales of variables are different and matter
MAX_FREQUENCY = 100 * 1000  # largest value in co-occurrence matrix
LOG_FREQUENCY = True  # take log of co-occurrence matrix element-wise

OFFSET = prep.midpoint

# ///////////////////////////////////////////////////////////////// TW matrix

# make term_by_window_co_occurrence_mat
start, end = 0, prep.store.num_tokens
tw_mat, xws, yws = make_term_by_context_co_occurrence_mat(
    prep, start=start, end=end, context_size=WINDOW_SIZE, max_frequency=MAX_FREQUENCY, log=LOG_FREQUENCY)


# ///////////////////////////////////////////////////////////////// LDA

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)
y_true = np.array([probe_store.cat2id[probe_store.probe2cat[w]] if w in probe_store.types else 0 for w in xws])
y_rand = np.random.permutation(y_true)

# reduce dimensionality
dims, _, _ = sparse_linalg.svds(tw_mat.T, k=NUM_DIMS, return_singular_vectors=True)

labels = [PROBES_NAME, PROBES_NAME + ' shuffled']
ys = [y_true, y_rand]
label2scores = {label: [] for label in labels}
for dim_id in range(NUM_DIMS):  # get columns cumulatively
    for label, y in zip(labels, ys):

        # prepare data for LDA (exclude non-probes)
        nonzero_ids = np.where(y != 0)[0]
        x = dims[nonzero_ids, :dim_id + 1]
        y = y[nonzero_ids]

        # fit and evaluate LDA
        print(f'Shape of input to LDA={x.shape}')
        clf = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                                         solver='svd', store_covariance=False)
        clf.fit(x, y)
        score = clf.score(x, y)
        print(f'dim={dim_id:>3} accuracy={score:.4f}')

        # collect
        label2scores[label].append(score)

    print()


_, ax = plt.subplots(dpi=192)
plt.title(f'window-size={WINDOW_SIZE}')
ax.set_ylabel('LDA Accuracy')
ax.set_xlabel('Number of Singular Dimensions in Training Data')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.set_ylim([0, 1.0])
# plot
x = np.arange(NUM_DIMS)
for label, y in label2scores.items():
    ax.plot(x, y, '-', label=label)
plt.legend()
plt.show()