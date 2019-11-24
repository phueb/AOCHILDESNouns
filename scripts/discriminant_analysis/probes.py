"""
Research questions:
1. How well can categories be distinguished in partition 1 vs. partition 2?
"""

import attr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.representation import make_context_by_term_matrix
from wordplay.memory import set_memory_limit

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-all'

docs = load_docs(CORPUS_NAME)
params = PrepParams(num_types=4096)  # TODO does limited vocabulary reproduce order effect?
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# /////////////////////////////////////////////////////////////////

CONTEXT_SIZE = 1

# ///////////////////////////////////////////////////////////////// co-occurrence matrix

tw_mat1, xws1, yws1 = make_context_by_term_matrix(
    prep.store.tokens, start=0, end=prep.midpoint, context_size=CONTEXT_SIZE)
tw_mat2, xws2, yws2 = make_context_by_term_matrix(
    prep.store.tokens, start=prep.midpoint, end=prep.store.num_tokens, context_size=CONTEXT_SIZE)

# ///////////////////////////////////////////////////////////////// LDA

set_memory_limit(prop=1.0)

# use only contexts common to both
common_yws = set(yws1).intersection(set(yws2))
print(f'Number of common contexts={len(common_yws)}')
row_ids1 = [yws1.index(yw) for yw in common_yws]
row_ids2 = [yws2.index(yw) for yw in common_yws]

# use only probes common to both
common_xws = set(xws1).intersection(set(xws2)).intersection(probe_store.types)
print(f'Number of common probes={len(common_xws)}')
col_ids1 = [xws1.index(xw) for xw in common_xws]
col_ids2 = [xws2.index(xw) for xw in common_xws]

# prepare x, y
x1 = tw_mat1.tocsr()[row_ids1].tocsc()[:, col_ids1].T.toarray()
x2 = tw_mat2.tocsr()[row_ids2].tocsc()[:, col_ids2].T.toarray()
y1 = [probe_store.cat2id[probe_store.probe2cat[p]] for p in common_xws]
y2 = [probe_store.cat2id[probe_store.probe2cat[p]] for p in common_xws]

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

    coefficients = clf.coef_.squeeze()  # ?
    print(coefficients.shape)
