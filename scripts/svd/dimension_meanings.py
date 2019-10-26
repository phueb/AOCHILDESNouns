from scipy.sparse import linalg as slinalg
from sklearn.preprocessing import normalize
from scipy import stats
import numpy as np
import attr
import matplotlib.pyplot as plt

from preppy.legacy import TrainPrep

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.pos import load_pos_words
from wordplay.svd import make_term_by_window_co_occurrence_mat

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'

SHUFFLE_DOCS = False
START_MID = False
START_END = False
TEST_FROM_MIDDLE = True  # put 1K docs in middle into test split

docs = load_docs(CORPUS_NAME,
                 test_from_middle=TEST_FROM_MIDDLE,
                 num_test_docs=0,
                 shuffle_docs=SHUFFLE_DOCS,
                 start_at_midpoint=START_MID,
                 start_at_ends=START_END)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

WINDOW_SIZE = 1
NUM_SVS = 16
NORMALIZE = False  # this makes all the difference - this means that the scales of variables are different and matter
MAX_FREQUENCY = 1000000  # largest value in co-occurrence matrix
LOG_FREQUENCY = False  # take log of co-occurrence matrix element-wise

MIN_FREQ = 30
POS = 'noun'
P_VALUE = 0.01 / NUM_SVS

# make term_by_window_co_occurrence_mats
start1, end1 = 0, prep.store.num_tokens
tw_mat1, xws1, yws1 = make_term_by_window_co_occurrence_mat(
    prep, start=start1, end=end1, window_size=WINDOW_SIZE, max_frequency=MAX_FREQUENCY, log=LOG_FREQUENCY)

mat = tw_mat1.asfptype().T  # transpose to do SVD, x-words now index rows
normalized = normalize(mat, axis=1, norm='l2', copy=False)
u, s, _ = slinalg.svds(normalized, k=NUM_SVS, return_singular_vectors=True)  # s is not 2D

pos_words = load_pos_words('childes-20180319-nouns')
print(f'Number of {POS}s after filtering={len(pos_words)}')


for pc_id in range(NUM_SVS):
    print()
    print('Singular Dim={} s={}'.format(NUM_SVS - pc_id, s[pc_id]))
    dimension = u[:, pc_id]
    sorted_ids = np.argsort(dimension)
    print([xws1[i] for i in sorted_ids[:+20] if prep.store.w2f[xws1[i]] > 10])
    print([xws1[i] for i in sorted_ids[-20:] if prep.store.w2f[xws1[i]] > 10])

    fig, ax = plt.subplots(1)
    # dimension = stats.norm.rvs(loc=0, scale=1, size=4096)
    stats.probplot(dimension, plot=ax)
    # plt.show()

    groups = [[v for v, w in zip(dimension, prep.store.types) if w in pos_words],
              [v for v, w in zip(dimension, prep.store.types) if w not in pos_words]]
    h, p = stats.kruskal(*groups)
    print(p)
    encodes_pos = p < P_VALUE
    print(f'Dimension encodes {POS}= {encodes_pos}')