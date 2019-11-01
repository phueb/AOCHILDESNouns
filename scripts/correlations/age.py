from scipy.stats import spearmanr
import numpy as np
import attr
from tabulate import tabulate

from preppy.legacy import TrainPrep

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.utils import split
from wordplay.pos import make_pos_words

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-4096'

SHUFFLE_DOCS = False
NUM_MID_TEST_DOCS = 0
NUM_PARTS = 2

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=NUM_MID_TEST_DOCS,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams(num_parts=NUM_PARTS)
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

POS_LIST = [
    'conjunction',
    'preposition',
    'noun',
    'verb',
    'adjective',
]

# TODO just iterate over tags text file?

pos_words_list = [make_pos_words(prep.store.types, pos) for pos in POS_LIST]

for tokens in split(prep.store.tokens, prep.num_tokens_in_part):

    for pos, pos_words in zip(POS_LIST, pos_words_list):
        y = len([w for w in tokens if tokens in pos_words])
        print(f'{pos:<16} num={y:,}')


raise SystemExit('Debugging')

# TODO don't just correlate with position in corpus - use the actual age value

# stats
rho_mat, p_mat = spearmanr(ao_features_mat)
print(p_mat < 0.05 / ao_features_mat.size)


data = [[1, 'Liquid', 24, 12],
        [2, 'Virtus.pro', 19, 14],
        [3, 'PSG.LGD', 15, 19],
        [4,'Team Secret', 10, 20]]
print (tabulate(data, headers=["Pos", "Team", "Win", "Lose"]))

