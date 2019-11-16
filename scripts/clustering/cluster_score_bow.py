"""
Research question:
1. Are semantic categories in part 1 of AO-CHILDES more easily clustered according to the gold structure?

There are two answers, depending on whether
* probe representations take into consideration word-order
* probe representations do not take into consideration word-order

This script computes BOW representations for probes

"""

from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import attr

from preppy.legacy import TrainPrep
from preppy.legacy import make_windows_mat
from categoryeval.probestore import ProbeStore
from categoryeval.score import calc_score

from wordplay import config
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.representation import make_bow_probe_representations
from wordplay.memory import set_memory_limit

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-4096'  # cannot use sem-all because some are too infrequent to occur in each partition

REVERSE = False
NUM_PARTS = 2
SHUFFLE_DOCS = False
CONTEXT_SIZE = 7

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=0,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE, context_size=CONTEXT_SIZE)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# ///////////////////////////////////////////////////////////////// parameters

DIRECTION = -1  # context is left if -1, context is right if +1  # -1
NORM = 'l1'  # l1
METRIC = 'f1'

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

part_ids = range(2)
part_id2score = {}
for part_id in part_ids:
    tokens = token_parts[part_id]
    # compute representations
    windows_mat = make_windows_mat(prep.reordered_parts[part_id],
                                   prep.num_windows_in_part,
                                   prep.num_tokens_in_window)
    probe_reps = make_bow_probe_representations(windows_mat,
                                                prep.store.types,
                                                probe_store.types,
                                                norm=NORM, direction=DIRECTION)

    # calc score
    score = calc_score(cosine_similarity(probe_reps), probe_store.gold_sims, metric=METRIC)

    # collect
    part_id2score[part_id] = score
    print('part_id={} score={:.3f}'.format(part_id, score))

# table
print('Semantic category information in AO-CHILDES'
      '\ncaptured by BOW representations\n'
      'with context-size={}'.format(CONTEXT_SIZE))
headers = ['Partition', METRIC]
rows = [(part_id + 1, part_id2score[part_id]) for part_id in part_ids]
print(tabulate(rows,
               headers=headers,
               tablefmt='latex',
               floatfmt=".4f"))