"""
Research question:
1. Are semantic categories in part 1 of AO-CHILDES more easily clustered according to the gold structure?

There are two answers, depending on whether
* probe representations take into consideration word-order
* probe representations do not take into consideration word-order

This script computes probe representations which take order into consideration

"""

from sklearn.metrics.pairwise import cosine_similarity
import attr
from tabulate import tabulate

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore
from categoryeval.score import calc_score

from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.representation import make_context_by_term_matrix
from wordplay.memory import set_memory_limit
from wordplay.location import make_w2locations

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-4096'  # careful: some probe reps might be zero vectors if they do not occur in part
NUM_PARTS = 2


docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=0,
                 num_test_take_random=0,
                 )

params = PrepParams(num_parts=NUM_PARTS, num_types=None)  # TODO
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# ///////////////////////////////////////////////////////////////// parameters

CONTEXT_SIZES = [1, 2, 3, 4]
METRIC = 'ck'
CONTROL_PROBE_DENSITY = True

if METRIC == 'ba':
    y_lims = [0.5, 1.0]
elif METRIC == 'ck':
    y_lims = [0.0, 0.2]
elif METRIC == 'f1':
    y_lims = [0.0, 0.2]
else:
    raise AttributeError('Invalid arg to "METRIC".')

# ///////////////////////////////////////////////////////////////// control for frequency

w2locations = make_w2locations(prep.store.tokens)
probe_locations = []
for w, locations in w2locations.items():
    if w in probe_store.types:
        probe_locations.extend(locations)

probes_median_location = np.median(probe_locations).astype(int)
print(f'{probes_median_location:,}')

# /////////////////////////////////////////////////////////////////

set_memory_limit(prop=0.9)

if CONTROL_PROBE_DENSITY:
    token_parts = [prep.store.tokens[:probes_median_location],
                   prep.store.tokens[-probes_median_location:]]
else:
    token_parts = [prep.store.tokens[:prep.midpoint],
                   prep.store.tokens[-prep.midpoint:]]

part_ids = range(2)
part_id2scores = {part_id: [] for part_id in part_ids}
for context_size in CONTEXT_SIZES:

    for part_id in part_ids:
        tokens = token_parts[part_id]

        # compute representations
        tw_mat, xws, yws = make_context_by_term_matrix(tokens,
                                                       context_size=context_size,
                                                       probe_store=probe_store)
        probe_reps = tw_mat.toarray().T  # transpose because representations are expected in the rows

        num_zero_rows = np.sum(~probe_reps.any(axis=1))
        print('num_zero_rows', num_zero_rows)

        try:
            score = calc_score(cosine_similarity(probe_reps), probe_store.gold_sims, metric=METRIC)
        except MemoryError:
            print(RuntimeWarning('Warning: Memory Error'))
            break  # print table before exiting

        # collect
        part_id2scores[part_id].append(score)
        print('part_id={} score={:.3f}'.format(part_id, score))

# table
print(f'{PROBES_NAME} category information in {CORPUS_NAME}\ncaptured by CT representations')
headers = ['Partition'] + CONTEXT_SIZES
rows = [['1'] + part_id2scores[0],
        ['2'] + part_id2scores[1]]
print(tabulate(rows,
               headers=headers,
               tablefmt='latex',
               floatfmt=".4f"))