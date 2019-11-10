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

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.representation import make_context_by_term_matrix
from wordplay.memory import set_memory_limit

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-4096'  # cannot use sem-all because some are too infrequent to occur in each partition

REVERSE = False
NUM_PARTS = 2
SHUFFLE_DOCS = False

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=0,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)

# ///////////////////////////////////////////////////////////////// parameters

NUM_EVALUATIONS = 2
CONTEXT_SIZES = [1, 2, 3, 4, 5, 6, 7]
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

token_parts = [prep.store.tokens[:prep.midpoint],
               prep.store.tokens[-prep.midpoint:]]

for context_size in CONTEXT_SIZES:
    part_ids = range(2)
    part_id2score = {}
    for part_id in part_ids:
        tokens = token_parts[part_id]

        # compute representations
        tw_mat, xws, yws = make_context_by_term_matrix(tokens,
                                                       context_size=context_size,
                                                       probe_store=probe_store)
        try:
            probe_reps = tw_mat.toarray().T  # transpose because representations are expected in the rows
        except MemoryError:
            raise SystemExit('Reached memory limit')

        # calc score
        score = calc_score(cosine_similarity(probe_reps), probe_store.gold_sims, metric=METRIC)

        # collect
        part_id2score[part_id] = score
        print('part_id={} score={:.3f}'.format(part_id, score))

    # table
    print('Semantic category information in AO-CHILDES'
          '\ncaptured by BOW representations\n'
          'with context-size={}'.format(context_size))
    headers = ['Partition', METRIC]
    rows = [(part_id + 1, part_id2score[part_id]) for part_id in part_ids]
    print(tabulate(rows,
                   headers=headers,
                   tablefmt='latex',
                   floatfmt=".4f"))
