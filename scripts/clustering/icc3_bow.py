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
import pingouin as pg
import pandas as pd
import numpy as np

from preppy.legacy import TrainPrep
from preppy.legacy import make_windows_mat
from categoryeval.probestore import ProbeStore

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

CONTEXT_SIZE = 2

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=0,
                 num_test_take_random=0,
                 )

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE, context_size=CONTEXT_SIZE)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# ///////////////////////////////////////////////////////////////// parameters

DIRECTION = -1  # context is left if -1, context is right if +1  # -1
NORM = 'l1'  # l1
MEASURE_NAME = 'ICC-3'

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

    ratings_p = cosine_similarity(probe_reps).flatten()  # predicted
    ratings_g = probe_store.gold_sims.astype(np.int).flatten()  # gold

    # make df for icc computation
    num_rows = len(ratings_p)
    target_col = list(np.tile(np.arange(len(probe_store.gold_sims)), len(probe_store.gold_sims.T))) * 2
    rater_col = [0 for _ in range(num_rows)] + [1 for _ in range(num_rows)]

    # iterate over thresholds, binarize similarities at each, and find best icc3
    scores = []
    NUM_THRESHOLDS = 100
    START_THRESHOLD = 0.9
    for thr in np.linspace(START_THRESHOLD, 1.0, NUM_THRESHOLDS):
        ratings_p_binary = np.array(ratings_p > thr).astype(np.int)
        rating_col = np.concatenate((ratings_p_binary, ratings_g))
        df = pd.DataFrame(data={'target': target_col, 'rater': rater_col, 'rating': rating_col})

        # TODO icc-3 computation returns nans - is implementation broken ?
        # calc icc-3, a measure of inter rater reliability
        icc = pg.intraclass_corr(data=df, targets='target', raters='rater', ratings='rating')

        score_at_thr = icc['ICC'][5]  # icc-3k is at index 5
        print(f'thr={thr:.2f} score={score_at_thr}')
        scores.append(score_at_thr)

        # number of elements that are equal
        print(np.sum(ratings_p_binary.astype(np.int) == ratings_g.astype(np.int)), len(ratings_p))
        print()

    # collect
    max_score = np.nanmax(scores)
    part_id2score[part_id] = max_score
    print('part_id={} score={:.3f}'.format(part_id, max_score))

# table
print('Semantic category information in AO-CHILDES'
      '\ncaptured by BOW representations\n'
      'with context-size={}'.format(CONTEXT_SIZE))
headers = ['Partition', MEASURE_NAME]
rows = [(part_id + 1, part_id2score[part_id]) for part_id in part_ids]
print(tabulate(rows,
               headers=headers,
               tablefmt='latex',
               floatfmt=".4f"))