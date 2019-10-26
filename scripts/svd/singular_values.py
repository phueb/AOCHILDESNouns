import matplotlib.pyplot as plt
from scipy.sparse import linalg as slinalg
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import attr

from preppy.legacy import TrainPrep

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.svd import make_term_by_window_co_occurrence_mat
from wordplay import config

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

WINDOW_SIZE = 2
NUM_SVS = 32
NORMALIZE = False  # this makes all the difference - this means that the scales of variables are different and matter
MAX_FREQUENCY = 1000000  # largest value in co-occurrence matrix
LOG_FREQUENCY = True  # take log of co-occurrence matrix element-wise

# make term_by_window_co_occurrence_mats
start1, end1 = 0, prep.midpoint // 1
start2, end2 = prep.store.num_tokens - end1, prep.store.num_tokens
label1 = 'partition 1' or 'tokens between\n{:,} & {:,}'.format(start1, end1)
label2 = 'partition 2' or 'tokens between\n{:,} & {:,}'.format(start2, end2)
tw_mat1, xws1, yws1 = make_term_by_window_co_occurrence_mat(
    prep, start=start1, end=end1, window_size=WINDOW_SIZE, max_frequency=MAX_FREQUENCY, log=LOG_FREQUENCY)
tw_mat2, xws2, yws2 = make_term_by_window_co_occurrence_mat(
    prep, start=start2, end=end2, window_size=WINDOW_SIZE, max_frequency=MAX_FREQUENCY, log=LOG_FREQUENCY)


# collect singular values
y1 = []
y2 = []
for y, mat in [(y1, tw_mat1.asfptype()),
               (y2, tw_mat2.asfptype())]:

    # compute variance of sparse matrix
    fit = StandardScaler(with_mean=False).fit(mat)
    print('sum of column variances of term-by-window co-occurrence matrix={:,}'.format(fit.var_.sum()))

    if NORMALIZE:
        print('Normalizing...')
        mat = normalize(mat, axis=1, norm='l2', copy=False)

    # SVD
    print('Fitting SVD ...')
    _, s, _ = slinalg.svds(mat, k=NUM_SVS, return_singular_vectors='vh')  # s is not 2D
    print('sum of singular values={:,}'.format(s.sum()))
    print('var of singular values={:,}'.format(s.var()))

    # collect singular values
    for sing_val in s[:-1][::-1]:  # last s is combination of all remaining s
        y.append(sing_val)
    print()

# figure
fig, ax = plt.subplots(1, figsize=(5, 5), dpi=None)
plt.title(f'SVD of AO-CHILDES partitions\nwindow size={WINDOW_SIZE}', fontsize=config.Fig.fontsize)
ax.set_ylabel('singular value', fontsize=config.Fig.fontsize)
ax.set_xlabel('Singular Dimension', fontsize=config.Fig.fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
ax.plot(y1, label=label1, linewidth=2)
ax.plot(y2, label=label2, linewidth=2)
ax.legend(loc='upper right', frameon=False, fontsize=config.Fig.fontsize)
plt.tight_layout()
plt.show()