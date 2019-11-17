"""
Research questions:
1. How discriminative of nouns are contexts in partition 1 vs. partition 2?
"""

import attr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.sparse.linalg as sparse_linalg

from preppy.legacy import TrainPrep

from wordplay import config
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.representation import make_context_by_term_matrix
from wordplay.pos import load_pos_words
from wordplay.memory import set_memory_limit

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'

docs = load_docs(CORPUS_NAME)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

CONTEXT_SIZE = 1
NUM_DIMS = None
NORMALIZE = False  # this makes all the difference - this means that the scales of variables are different and matter


OFFSET = prep.midpoint

# ///////////////////////////////////////////////////////////////// TW matrix

# make term_by_window_co_occurrence_mats
start1, end1 = 0, OFFSET
start2, end2 = prep.store.num_tokens - OFFSET, prep.store.num_tokens
tw_mat1, xws1, yws1 = make_context_by_term_matrix(
    prep.store.tokens, start=start1, end=end1, context_size=CONTEXT_SIZE)
tw_mat2, xws2, yws2 = make_context_by_term_matrix(
    prep.store.tokens, start=start2, end=end2, context_size=CONTEXT_SIZE)


# ///////////////////////////////////////////////////////////////// LDA

set_memory_limit(prop=1.0)

nouns = load_pos_words(f'{CORPUS_NAME}-nouns')

# use only features common to both
common_yws = set(yws1).intersection(set(yws2))
print(f'Number of common contexts={len(common_yws)}')
x1_common = tw_mat1.T[:, [n for n, yw in enumerate(yws1) if yw in common_yws]]
x2_common = tw_mat2.T[:, [n for n, yw in enumerate(yws2) if yw in common_yws]]
y1 = [1 if w in nouns else 0 for w in xws1]
y2 = [1 if w in nouns else 0 for w in xws2]

if NUM_DIMS is not None:
    # reduce dimensionality
    x1, _, _ = sparse_linalg.svds(x1_common, k=NUM_DIMS, return_singular_vectors=True)
    x2, _, _ = sparse_linalg.svds(x2_common, k=NUM_DIMS, return_singular_vectors=True)
else:
    x1 = x1_common.toarray()
    x2 = x2_common.toarray()

for x, y in zip([x1, x2],
                [y1, y2]):

    print(f'Shape of input to LDA={x.shape}')
    clf = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                                     solver='svd', store_covariance=False)
    try:
        clf.fit(x, y)

    except MemoryError as e:
        raise SystemExit('Reached memory limit')

    # how well does discriminant function work for other part?
    score1 = clf.score(x1, y1)
    score2 = clf.score(x2, y2)
    print(f'partition-1 accuracy={score1:.3f}')
    print(f'partition-2 accuracy={score2:.3f}')

    coefficients = clf.coef_.squeeze()
    s = sorted(zip(common_yws, coefficients), key=lambda i: i[1])  # TODO not sure about this
    print(s[:10])
    print(s[-10:])