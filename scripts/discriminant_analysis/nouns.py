"""
Research questions:
1. How discriminative of nouns are contexts in partition 1 vs. partition 2?
"""

import attr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from preppy.legacy import TrainPrep

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.svd import make_term_by_window_co_occurrence_mat
from wordplay.pos import load_pos_words

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

WINDOW_SIZE = 3
NORMALIZE = False  # this makes all the difference - this means that the scales of variables are different and matter
MAX_FREQUENCY = 100 * 1000  # largest value in co-occurrence matrix
LOG_FREQUENCY = True  # take log of co-occurrence matrix element-wise

OFFSET = 1000 * 1000

# ///////////////////////////////////////////////////////////////// TW matrix

# make term_by_window_co_occurrence_mats
start1, end1 = 0, OFFSET
start2, end2 = prep.store.num_tokens - OFFSET, prep.store.num_tokens
tw_mat1, xws1, yws1 = make_term_by_window_co_occurrence_mat(
    prep, start=start1, end=end1, window_size=WINDOW_SIZE, max_frequency=MAX_FREQUENCY, log=LOG_FREQUENCY)
tw_mat2, xws2, yws2 = make_term_by_window_co_occurrence_mat(
    prep, start=start2, end=end2, window_size=WINDOW_SIZE, max_frequency=MAX_FREQUENCY, log=LOG_FREQUENCY,)

# ///////////////////////////////////////////////////////////////// LDA

nouns = load_pos_words(f'{CORPUS_NAME}-nouns+plurals')

for mat, xws in zip([tw_mat1, tw_mat2],
                    [xws1, xws2]):

    x = mat.T.toarray()
    y = [1 if w in nouns else 0 for w in xws]
    print(f'Number of words included in LDA={len(y)}')
    clf = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                                     solver='svd', store_covariance=False)
    clf.fit(x, y)
    score = clf.score(x, y)
    print(score)
    print(f'prop var of comp1: {clf.explained_variance_ratio_[0]:.2f}')
