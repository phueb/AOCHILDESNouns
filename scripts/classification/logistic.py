"""
Research questions:
1. How well can categories be distinguished in partition 1 vs. partition 2?
"""

import attr
from sklearn.linear_model import LogisticRegression
from sortedcontainers import SortedSet

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.representation import make_context_by_term_matrix
from wordplay.memory import set_memory_limit

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'syn-4096'

docs = load_docs(CORPUS_NAME)
params = PrepParams(num_types=None)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# /////////////////////////////////////////////////////////////////

CONTEXT_SIZE = 3

# ///////////////////////////////////////////////////////////////// co-occurrence matrix

tw_mat1, xws1, yws1 = make_context_by_term_matrix(
    prep.store.tokens,
    start=0,
    end=prep.midpoint,
    context_size=CONTEXT_SIZE,
    probe_store=probe_store)

tw_mat2, xws2, yws2 = make_context_by_term_matrix(
    prep.store.tokens,
    start=prep.midpoint,
    end=prep.store.num_tokens,
    context_size=CONTEXT_SIZE,
    probe_store=probe_store)

# /////////////////////////////////////////////////////////////////

set_memory_limit(prop=1.0)

# use only contexts common to both and contexts that were actually collected
common_yws = SortedSet(set(yws1).intersection(set(yws2)))
common_yws = [yw for yw in common_yws if
              yws1.index(yw) < tw_mat1.shape[0] and
              yws2.index(yw) < tw_mat2.shape[0]]

print(f'Number of common contexts={len(common_yws)}')
row_ids1 = [yws1.index(yw) for yw in common_yws]
row_ids2 = [yws2.index(yw) for yw in common_yws]

# use only probes common to both
col_ids1 = [xws1.index(xw) for xw in probe_store.types]
col_ids2 = [xws2.index(xw) for xw in probe_store.types]

# prepare x, y
x1 = tw_mat1.tocsr()[row_ids1].tocsc()[:, col_ids1].T.toarray()
x2 = tw_mat2.tocsr()[row_ids2].tocsc()[:, col_ids2].T.toarray()
y1 = [probe_store.cat2id[probe_store.probe2cat[p]] for p in probe_store.types]
y2 = [probe_store.cat2id[probe_store.probe2cat[p]] for p in probe_store.types]

for x, y in zip([x1, x2],
                [y1, y2]):

    print(f'Shape of input to classifier={x.shape}')
    clf = LogisticRegression(C=1,
                             penalty='l1',
                             # solver='saga',
                             multi_class='ovr')
    try:
        clf.fit(x, y)

    except MemoryError as e:
        raise SystemExit('Reached memory limit')

    print(f'{clf.score(x, y):.3f}')

    # score on both partitions
    # score1 = clf.score(x1, y1)
    # score2 = clf.score(x2, y2)
    # print(f'partition-1 accuracy={score1:.3f}')
    # print(f'partition-2 accuracy={score2:.3f}')

    coefficients = clf.coef_  # has shape (num discriminant fns, num features)
    # print(coefficients)
    # print(coefficients.shape)
