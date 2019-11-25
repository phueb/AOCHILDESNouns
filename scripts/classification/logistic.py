"""
Research questions:
1. How well can categories be distinguished in partition 1 vs. partition 2?
"""

import attr
from sklearn.linear_model import LogisticRegression
from sortedcontainers import SortedSet
from sortedcontainers import SortedDict

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.utils import get_sliding_windows
from wordplay.memory import set_memory_limit
from wordplay.representation import make_probe_reps_median_split

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-all'

docs = load_docs(CORPUS_NAME)
params = PrepParams(num_types=4096)  # TODO num types
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# /////////////////////////////////////////////////////////////////

CONTEXT_SIZE = 4

# ///////////////////////////////////////////////////////////////// representations

# get all probe contexts
probe2contexts = SortedDict({p: [] for p in probe_store.types})
contexts_in_order = get_sliding_windows(CONTEXT_SIZE, prep.store.tokens)
y_words = SortedSet(contexts_in_order)
yw2row_id = {c: n for n, c in enumerate(y_words)}
context_types = SortedSet()
for n, context in enumerate(contexts_in_order[:-CONTEXT_SIZE]):
    # update probe2contexts
    next_context = contexts_in_order[n + 1]
    target = next_context[-1]
    if target in probe_store.types:
        probe2contexts[target].append(context)
        # update context types
        context_types.add(context)

x1 = make_probe_reps_median_split(probe2contexts, context_types, split_id=0)
x2 = make_probe_reps_median_split(probe2contexts, context_types, split_id=1)

# /////////////////////////////////////////////////////////////////

set_memory_limit(prop=1.0)

# prepare y
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

    # score on both partitions
    score1 = clf.score(x1, y1)
    score2 = clf.score(x2, y2)
    print(f'partition-1 accuracy={score1:.3f}')
    print(f'partition-2 accuracy={score2:.3f}')
