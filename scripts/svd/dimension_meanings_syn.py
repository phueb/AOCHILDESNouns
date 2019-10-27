from scipy.sparse import linalg as slinalg
from sklearn.preprocessing import normalize
from scipy import stats
import numpy as np
import attr
import seaborn as sns
import matplotlib.pyplot as plt

from preppy.legacy import TrainPrep

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.pos import load_pos_words
from wordplay.svd import make_term_by_window_co_occurrence_mat
from wordplay.svd import inspect_loadings
from wordplay import config

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-4096'

SHUFFLE_DOCS = False
TEST_FROM_MIDDLE = True  # put 1K docs in middle into test split

docs = load_docs(CORPUS_NAME,
                 test_from_middle=TEST_FROM_MIDDLE,
                 num_test_docs=0)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

WINDOW_SIZE = 2
NUM_SVS = 256
NORMALIZE = True
MAX_FREQUENCY = 1000 * 100  # largest value in co-occurrence matrix
LOG_FREQUENCY = False  # take log of co-occurrence matrix element-wise

ALPHA = 0.001 / NUM_SVS

PLOT_LOADINGS = False

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

# make categories for probing
names = ('nouns', 'verbs', 'pronouns', 'random')
name2words = {}
for name in names:
    category_words = load_pos_words(f'{CORPUS_NAME}-{name}')
    name2words[name] = category_words
    print(f'Loaded {len(category_words)} words in category {name}')
    assert len(category_words) > 0

# collect p-value for each singular dimension for plotting
name2ps = {name: [] for name in names}
for pc_id in range(NUM_SVS):
    print()
    print(f'Singular Dimension={NUM_SVS - pc_id} singular value={s[pc_id]}')
    dimension = u[:, pc_id]

    # print highest and lowest scoring words
    sorted_ids = np.argsort(dimension)
    print([xws1[i] for i in sorted_ids[:+20]])
    print([xws1[i] for i in sorted_ids[-20:]])

    for name in names:
        category_words = name2words[name]

        # non-parametric analysis of variance.
        # is variance between category words and random words different?
        groups = [[v for v, w in zip(dimension, prep.store.types) if w in category_words],
                  [v for v, w in zip(dimension, prep.store.types) if w not in category_words]]
        _, p = stats.kruskal(*groups)
        print(p)
        print(f'Dimension encodes {name}= {p < ALPHA}')
        name2ps[name].append(p)

        # inspect how category words actually differ in their loadings from other words
        if p < ALPHA and PLOT_LOADINGS:
            inspect_loadings(prep, dimension, category_words, name2words['random'])


# a dimension cannot encode both nouns and verbs - so chose best
name2y = {name: [] for name in names}
for ps_at_sd in zip(*[name2ps[name] for name in names]):
    values = np.array(ps_at_sd)  # allows item assignment
    bool_ids = np.where(values < ALPHA)[0]

    # in case the dimension encodes more than 1 category, only allow 1 winner
    # by setting all but lowest value to np.nan
    if len(bool_ids) > 1:
        min_i = np.argmin(ps_at_sd).item()
        ps_at_sd = [v if i == min_i else np.nan for i, v in enumerate(ps_at_sd)]
        print(f'WARNING: Dimension encodes multiple categories')

    # collect
    for n, name in enumerate(names):
        name2y[name].append(0.02 * n if values[n] < ALPHA else np.nan)

# figure
_, ax = plt.subplots(dpi=192, figsize=(6, 6))
ax.set_title(f'Decoding Singular Dimensions\nof term-by-window matrix\nwindow size={WINDOW_SIZE}',
             fontsize=config.Fig.fontsize)
# axis
ax.set_xlabel('Singular Dimension', fontsize=config.Fig.fontsize)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False, left=False)
ax.set_yticks([])
ax.set_xlim(left=0, right=NUM_SVS)

# scatter
x = np.arange(NUM_SVS)
num_categories = len(names)
name2color = {n: c for n, c in zip(names, sns.color_palette("hls", num_categories))}
for name in names:
    ax.scatter(x, name2y[name][::-1], color=name2color[name], label=name)

plt.legend(frameon=True, framealpha=1.0, bbox_to_anchor=(0.5, 1.4), ncol=4, loc='center')
plt.show()

