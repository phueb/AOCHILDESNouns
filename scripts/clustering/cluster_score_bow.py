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

from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.representation import make_bow_probe_representations
from wordplay.memory import set_memory_limit

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'syn-nva'  # cannot use sem-all because some are too infrequent to occur in each partition
NUM_PARTS = 2


docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=0,
                 num_test_take_random=0,
                 )

params = PrepParams(num_parts=NUM_PARTS)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# ///////////////////////////////////////////////////////////////// parameters

DIRECTION = -1  # context is left if -1, context is right if +1  # -1
NORM = 'l1'  # l1
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

part_ids = range(2)
part_id2scores = {part_id: [] for part_id in part_ids}
for context_size in CONTEXT_SIZES:

    for part_id in part_ids:

        # overwrite context_size
        prep.context_size = context_size

        # compute representations
        windows_mat = make_windows_mat(prep.reordered_parts[part_id],
                                       prep.num_windows_in_part,
                                       prep.num_tokens_in_window)
        probe_reps = make_bow_probe_representations(windows_mat,
                                                    prep.store.types,
                                                    probe_store.types,
                                                    norm=NORM, direction=DIRECTION)
        try:
            score = calc_score(cosine_similarity(probe_reps), probe_store.gold_sims, metric=METRIC)
        except MemoryError:
            break  # print table before exiting

        # collect
        part_id2scores[part_id].append(score)
        print('part_id={} score={:.3f}'.format(part_id, score))

# table
print(f'{PROBES_NAME} category information in {CORPUS_NAME}\ncaptured by BOW representations')
# noinspection PyTypeChecker
headers = ['Partition'] + CONTEXT_SIZES
rows = [['1'] + part_id2scores[0],
        ['2'] + part_id2scores[1]]
print(tabulate(rows,
               headers=headers,
               tablefmt='latex',
               floatfmt=".4f"))