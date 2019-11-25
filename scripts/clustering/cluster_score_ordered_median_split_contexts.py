"""
Research question:
1. Are semantic categories in part 1 of AO-CHILDES more easily clustered according to the gold structure?

There are two answers, depending on whether
* probe representations take into consideration word-order
* probe representations do not take into consideration word-order

This script computes probe representations which take order into consideration


this script is probably the fairest way to compare performance between partitions because
it ensures that for each probe, the same number of contexts occur in each "partition"
(the word "partitions" doesn't really apply here, as the corpus isn't split, but contexts are split)

"""
from sklearn.metrics.pairwise import cosine_similarity
import attr
from tabulate import tabulate
from sortedcontainers import SortedSet

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore
from categoryeval.score import calc_score

from wordplay.representation import make_probe_reps_median_split
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.utils import get_sliding_windows
from wordplay.memory import set_memory_limit

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-all'  # careful: some probe reps might be zero vectors if they do not occur in part


docs = load_docs(CORPUS_NAME)

params = PrepParams(num_types=None)  # TODO
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# ///////////////////////////////////////////////////////////////// parameters

CONTEXT_SIZES = [1, 2, 3, 4]
METRIC = 'ck'

if METRIC == 'ba':
    y_lims = [0.5, 1.0]
elif METRIC == 'ck':
    y_lims = [0.0, 0.2]
elif METRIC == 'f1':
    y_lims = [0.0, 0.2]
else:
    raise AttributeError('Invalid arg to "METRIC".')

# /////////////////////////////////////////////////////////////////

set_memory_limit(prop=0.9)

split_ids = range(2)
split_id2scores = {split_id: [] for split_id in split_ids}

for context_size in CONTEXT_SIZES:

    # get all probe contexts
    probe2contexts = {p: [] for p in probe_store.types}
    contexts_in_order = get_sliding_windows(context_size, prep.store.tokens)
    y_words = SortedSet(contexts_in_order)
    yw2row_id = {c: n for n, c in enumerate(y_words)}
    context_types = SortedSet()
    for n, context in enumerate(contexts_in_order[:-context_size]):
        # update probe2contexts
        next_context = contexts_in_order[n + 1]
        target = next_context[-1]
        if target in probe_store.types:
            probe2contexts[target].append(context)
            # update context types
            context_types.add(context)

    for split_id in split_ids:

        probe_reps = make_probe_reps_median_split(probe2contexts, context_types, split_id)

        try:
            score = calc_score(cosine_similarity(probe_reps), probe_store.gold_sims, metric=METRIC)
        except MemoryError:
            print(RuntimeWarning('Warning: Memory Error'))
            break  # print table before exiting

        # collect
        split_id2scores[split_id].append(score)
        print('split_id={} score={:.3f}'.format(split_id, score))

# table
print(f'{PROBES_NAME} category information in {CORPUS_NAME}\ncaptured by CT representations')
headers = ['Split'] + CONTEXT_SIZES
rows = [['1'] + split_id2scores[0],
        ['2'] + split_id2scores[1]]
print(tabulate(rows,
               headers=headers,
               tablefmt='latex',
               floatfmt=".4f"))