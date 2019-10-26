from scipy.sparse import linalg as slinalg
from sklearn.preprocessing import normalize
from scipy import stats
import numpy as np
import attr
from matplotlib.axes import Axes
from matplotlib import gridspec
import matplotlib.pyplot as plt

from preppy.legacy import TrainPrep

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.pos import load_pos_words, make_pos_words
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

WINDOW_SIZE = 3
NUM_SVS = 128
NORMALIZE = True  # this makes all the difference - this means that the scales of variables are different and matter
MAX_FREQUENCY = 1000  # largest value in co-occurrence matrix
LOG_FREQUENCY = True  # take log of co-occurrence matrix element-wise

ALPHA = 0.01 / NUM_SVS

PLOT_QQ = False

# make term_by_window_co_occurrence_mats
start1, end1 = 0, prep.store.num_tokens
tw_mat1, xws1, yws1 = make_term_by_window_co_occurrence_mat(
    prep, start=start1, end=end1, window_size=WINDOW_SIZE, max_frequency=MAX_FREQUENCY, log=LOG_FREQUENCY)
mat = tw_mat1.T.asfptype()  # transpose so that x-words now index rows (does not affect singular values)

# normalization
if NORMALIZE:
    mat = normalize(mat, axis=1, norm='l2', copy=False)

# svd
u, s, _ = slinalg.svds(mat, k=NUM_SVS, return_singular_vectors=True)  # s is not 2D

nouns = load_pos_words('childes-20180319-nouns')
verbs = load_pos_words('childes-20180319-verbs')
rands = np.random.choice(prep.store.types, size=len(nouns), replace=False)
prons = ['me', 'you', 'he', 'they', 'she', 'it', 'we']
print(f'Loaded {len(rands)} rands ')
print(f'Loaded {len(nouns)} nouns ')
print(f'Loaded {len(verbs)} verbs ')
print(f'Loaded {len(prons)} prons ')

# collect (1-p) for each singular dimension for plotting
rand_ps = []
noun_ps = []
verb_ps = []
pron_ps = []
for pc_id in range(NUM_SVS):
    print()
    print('Singular Dim={} s={}'.format(NUM_SVS - pc_id, s[pc_id]))
    dimension = u[:, pc_id]
    sorted_ids = np.argsort(dimension)
    print([xws1[i] for i in sorted_ids[:+20] if prep.store.w2f[xws1[i]] > 10])
    print([xws1[i] for i in sorted_ids[-20:] if prep.store.w2f[xws1[i]] > 10])

    # qq plot shows that singular dimensions have non-normal distribution.
    # this means that  non-parametric test must be used to decode meaning (e.g. not point biserial correlation)
    if PLOT_QQ:
        fig, axes = plt.subplots(1)
        dimension = stats.norm.rvs(loc=0, scale=1, size=4096)
        stats.probplot(dimension, plot=axes)
        plt.show()

    # non-parametric analysis of variance.
    # is variance between random words  different?
    groups = [[v for v, w in zip(dimension, prep.store.types) if w in rands],
              [v for v, w in zip(dimension, prep.store.types) if w not in rands]]
    _, p = stats.kruskal(*groups)
    print(p)
    print(f'Dimension encodes rands= {p < ALPHA}')
    rand_ps.append(p)

    # non-parametric analysis of variance.
    # is variance between nouns and non-nouns different?
    groups = [[v for v, w in zip(dimension, prep.store.types) if w in nouns],
              [v for v, w in zip(dimension, prep.store.types) if w not in nouns]]
    _, p = stats.kruskal(*groups)
    print(p)
    print(f'Dimension encodes nouns= {p < ALPHA}')
    noun_ps.append(p)

    # non-parametric analysis of variance.
    # is variance between verbs and non-verbs different?
    groups = [[v for v, w in zip(dimension, prep.store.types) if w in verbs],
              [v for v, w in zip(dimension, prep.store.types) if w not in verbs]]
    _, p = stats.kruskal(*groups)
    print(p)
    print(f'Dimension encodes verbs= {p < ALPHA}')
    verb_ps.append(p)

    # non-parametric analysis of variance.
    # is variance between pronouns and non-pronouns different?
    groups = [[v for v, w in zip(dimension, prep.store.types) if w in prons],
              [v for v, w in zip(dimension, prep.store.types) if w not in prons]]
    _, p = stats.kruskal(*groups)
    print(p)
    print(f'Dimension encodes verbs= {p < ALPHA}')
    pron_ps.append(p)


# figure
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
ax0: Axes = plt.subplot(gs[0])
ax1: Axes = plt.subplot(gs[1])
ax0.set_title(f'Decoding Singular Dimensions\nof term-by-window matrix\nwindow size={WINDOW_SIZE}',
              fontsize=config.Fig.fontsize)
x = np.arange(NUM_SVS)
y0 = rand_ps[::-1]
y1 = noun_ps[::-1]
y2 = verb_ps[::-1]
y3 = pron_ps[::-1]

# a dimension cannot encode both nouns and verbs - so chose best
y02 = []
y12 = []
y22 = []
y32 = []
for values in zip(y0, y1, y2, y3):
    values = list(values)  # allows item assignment
    if values[0] < ALPHA and values[1] < ALPHA:
        i = np.argmax(values).item()
        values[i] = np.nan  # set largest value to np.nan
        print(f'WARNING: Dimension encodes multiple categories')

    y02.append(0.00 if values[0] < ALPHA else np.nan)
    y12.append(0.02 if values[1] < ALPHA else np.nan)
    y22.append(0.04 if values[2] < ALPHA else np.nan)
    y32.append(0.06 if values[3] < ALPHA else np.nan)

# axis 0
ax0.set_ylabel('1-p', fontsize=config.Fig.fontsize)
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.tick_params(axis='both', which='both', top=False, right=False)
ax0.set_xticks([])
ax0.scatter(x, 1 - np.array(y0), zorder=1, color='grey')
ax0.scatter(x, 1 - np.array(y1), zorder=1, color='grey')
ax0.scatter(x, 1 - np.array(y2), zorder=1, color='grey')
ax0.scatter(x, 1 - np.array(y3), zorder=1, color='grey')

# axis 1
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.tick_params(axis='both', which='both', top=False, right=False, left=False)
ax1.set_yticks([])
ax1.scatter(x, y02, color='blue', label='rands')
ax1.scatter(x, y12, color='red', label='nouns')
ax1.scatter(x, y22, color='green', label='verbs')
ax1.scatter(x, y32, color='orange', label='pronouns')
ax1.set_xlabel('Singular Dimension', fontsize=config.Fig.fontsize)

plt.legend(frameon=True, framealpha=1.0)
plt.subplots_adjust(top=0.5)
plt.show()

