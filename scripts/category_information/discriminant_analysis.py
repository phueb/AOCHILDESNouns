"""
Research questions:
1. How discriminative of semantic categories are probe contexts?
"""

import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import attr
import numpy as np
import pyprind
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.svd import make_term_by_window_co_occurrence_mat

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-4096'

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

WINDOW_SIZE = 1
NUM_DIMS = 256
NORMALIZE = False  # this makes all the difference - this means that the scales of variables are different and matter
MAX_FREQUENCY = 100 * 1000  # largest value in co-occurrence matrix
LOG_FREQUENCY = True  # take log of co-occurrence matrix element-wise

OFFSET = prep.midpoint


# ///////////////////////////////////////////////////////////////// TW matrix

# make term_by_window_co_occurrence_mats
start1, end1 = 0, OFFSET
start2, end2 = prep.store.num_tokens - OFFSET, prep.store.num_tokens
tw_mat1, xws1, yws1 = make_term_by_window_co_occurrence_mat(
    prep, start=start1, end=end1, window_size=WINDOW_SIZE, max_frequency=MAX_FREQUENCY, log=LOG_FREQUENCY,
    probe_store=probe_store)
tw_mat2, xws2, yws2 = make_term_by_window_co_occurrence_mat(
    prep, start=start2, end=end2, window_size=WINDOW_SIZE, max_frequency=MAX_FREQUENCY, log=LOG_FREQUENCY,
    probe_store=probe_store)

# ///////////////////////////////////////////////////////////////// LDA

x1 = tw_mat1.T.toarray()
y1 = [probe_store.cat2id[probe_store.probe2cat[p]] for p in xws1]
clf = LinearDiscriminantAnalysis()
clf.fit(x1, y1)
score1 = clf.score(x1, y1)
print(score1)

x2 = tw_mat2.T.toarray()
y2 = [probe_store.cat2id[probe_store.probe2cat[p]] for p in xws2]
clf = LinearDiscriminantAnalysis()
clf.fit(x2, y2)
score2 = clf.score(x2, y2)
print(score2)