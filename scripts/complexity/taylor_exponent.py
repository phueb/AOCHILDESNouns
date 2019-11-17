"""
Research questions:
1. Is language in partition 1 more systematic or template-like?
"""

import numpy as np
from collections import Counter
from scipy import optimize
import matplotlib.pyplot as plt
import attr

from preppy.legacy import TrainPrep

from wordplay import config
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-all'


NUM_MID_TEST_DOCS = 0
NUM_PARTS = 2
NUM_TYPES = 1000 * 26

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=NUM_MID_TEST_DOCS,
                 num_test_take_random=0,
                 )

params = PrepParams(num_parts=NUM_PARTS, num_types=NUM_TYPES)
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////


SPLIT_SIZE = 5620
CORPUS_ID = 1  # 0=mobydick, 1=CHILDES
PLOT_FIT = False


def fitfunc(p, x):
    return p[0] + p[1] * x


def errfunc(p, x, y):
    return y - fitfunc(p, x)


corpus_name = ['mobydick', 'childes-20180319'][CORPUS_ID]


for part_id, part in enumerate(prep.reordered_parts):
    # make freq_mat
    num_splits = prep.num_tokens_in_part // SPLIT_SIZE + 1
    freq_mat = np.zeros((prep.store.num_types, num_splits))
    start_locs = np.arange(0, prep.num_tokens_in_part, SPLIT_SIZE)
    num_start_locs = len(start_locs)
    for split_id, start_loc in enumerate(start_locs):
        for token_id, f in Counter(part[start_loc:start_loc + SPLIT_SIZE]).items():
            freq_mat[token_id, split_id] = f
    # x, y
    freq_mat = freq_mat[~np.all(freq_mat == 0, axis=1)]
    x = freq_mat.mean(axis=1)  # make sure not to have rows with zeros
    y = freq_mat.std(axis=1)
    # fit
    pinit = np.array([1.0, -1.0])
    logx = np.log10(x)
    logy = np.log10(y)
    out = optimize.leastsq(errfunc, pinit, args=(logx, logy), full_output=True)

    for i in out:
        print(i)

    pfinal = out[0]
    amp = pfinal[0]
    alpha = pfinal[1]
    # fig
    fig, ax = plt.subplots()
    plt.title(f'{corpus_name}\nnum_types={NUM_TYPES:,}, part {part_id + 1} of {NUM_PARTS}')
    ax.set_xlabel('mean')
    ax.set_ylabel('std')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # plot
    ax.text(x=1.0, y=0.3, s='Taylor\'s exponent: {:.3f}'.format(alpha))
    ax.loglog(x, y, '.', markersize=2)
    if PLOT_FIT:
        ax.loglog(x, amp * (x ** alpha) + 0, '.', markersize=2)  # TODO test
    plt.show()






