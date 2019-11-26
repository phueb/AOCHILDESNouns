"""
Research questions:
1. How well can categories be distinguished in partition 1 vs. partition 2?
"""

import attr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
params = PrepParams(num_types=None)  # TODO num types
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# /////////////////////////////////////////////////////////////////

ORDERED_REPRESENTATIONS = True
CONTEXT_SIZE = 4

# ///////////////////////////////////////////////////////////////// representations

# get all probe contexts
probe2contexts = SortedDict({p: [] for p in probe_store.types})
contexts_in_order = get_sliding_windows(CONTEXT_SIZE, prep.store.tokens)
y_words = SortedSet(contexts_in_order)
yw2row_id = {c: n for n, c in enumerate(y_words)}
context_types = SortedSet()
for n, context in enumerate(contexts_in_order[:-CONTEXT_SIZE]):
    next_context = contexts_in_order[n + 1]
    target = next_context[-1]
    if target in probe_store.types:

        if ORDERED_REPRESENTATIONS:
            probe2contexts[target].append(context)
            context_types.add(context)
        else:
            single_word_contexts = [(w,) for w in context]
            probe2contexts[target].extend(single_word_contexts)
            context_types.update(single_word_contexts)

x1 = make_probe_reps_median_split(probe2contexts, context_types, split_id=0)
x2 = make_probe_reps_median_split(probe2contexts, context_types, split_id=1)

# note: LDA classifier appears to use l2 normalization internally
# because results are same if normalization is performed externally

# /////////////////////////////////////////////////////////////////

set_memory_limit(prop=1.0)

# prepare y
y1 = [probe_store.cat2id[probe_store.probe2cat[p]] for p in probe_store.types]
y2 = [probe_store.cat2id[probe_store.probe2cat[p]] for p in probe_store.types]

for x, y in zip([x1, x2],
                [y1, y2]):

    print(f'Shape of input to classifier={x.shape}')
    clf = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                                     solver='svd', store_covariance=False)
    try:
        clf.fit(x, y)

    except MemoryError as e:
        raise SystemExit('Reached memory limit')

    # mean-accuracy
    score = clf.score(x, y)
    print(f'accuracy={score:.3f}')

print('CONTEXT_SIZE', CONTEXT_SIZE)
print('ORDERED', ORDERED_REPRESENTATIONS)