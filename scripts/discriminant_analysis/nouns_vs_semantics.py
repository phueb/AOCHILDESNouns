"""
Research questions:
1.Which singular dimensions of input result in good noun discrimination
 and which result in good semantic category discrimination?
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
from wordplay.svd import make_term_by_window_co_occurrence_mat
from wordplay.pos import load_pos_words

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
SEM_PROBES_NAME = 'sem-4096'
SYN_PROBES_NAME = 'syn-4096'  # TODO make one with only verbs vs. nouns vs. adjectives

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
NUM_DIMS = 512
NORMALIZE = False  # this makes all the difference - this means that the scales of variables are different and matter
MAX_FREQUENCY = 100 * 1000  # largest value in co-occurrence matrix
LOG_FREQUENCY = True  # take log of co-occurrence matrix element-wise

OFFSET = prep.midpoint

# ///////////////////////////////////////////////////////////////// TW matrix

# make term_by_window_co_occurrence_mat
start, end = 0, prep.store.num_tokens
tw_mat, xws, yws = make_term_by_window_co_occurrence_mat(
    prep, start=start, end=end, window_size=WINDOW_SIZE, max_frequency=MAX_FREQUENCY, log=LOG_FREQUENCY)


# ///////////////////////////////////////////////////////////////// LDA

syn_store = ProbeStore(CORPUS_NAME, SYN_PROBES_NAME, prep.store.w2id)
sem_store = ProbeStore(CORPUS_NAME, SEM_PROBES_NAME, prep.store.w2id)
y_syn = np.array([syn_store.cat2id[syn_store.probe2cat[w]] if w in syn_store.types else 0 for w in xws])
y_sem = np.array([sem_store.cat2id[sem_store.probe2cat[w]] if w in sem_store.types else 0 for w in xws])

# reduce dimensionality
dims, _, _ = sparse_linalg.svds(tw_mat.T, k=NUM_DIMS, return_singular_vectors=True)

labels = ['syntactic', 'semantic']
label2scores = {label: [] for label in labels}
for dim_id in range(NUM_DIMS):  # get columns cumulatively
    for label, y in zip(labels, [y_syn, y_sem]):

        # prepare data for LDA
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
plt.title('')
ax.set_ylabel('LDA Accuracy')
ax.set_xlabel('Number of Singular Dimensions in Training Data')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
x = np.arange(NUM_DIMS)
for label, y in label2scores.items():
    ax.plot(x, y, '-', label=label)
plt.legend()
plt.show()