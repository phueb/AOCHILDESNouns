import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import attr

from preppy.legacy import TrainPrep
from preppy.legacy import make_windows_mat
from categoryeval.probestore import ProbeStore

from wordplay.params import PrepParams
from wordplay.docs import load_docs

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-4096'

SHUFFLE_DOCS = False
NUM_MID_TEST_DOCS = 100

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=NUM_MID_TEST_DOCS,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams(num_parts=2)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)

# /////////////////////////////////////////////////////////////////

WINDOW_SIZES = [2, 3, 4, 5, 6]

windows_mat1 = make_windows_mat(prep.reordered_halves[0], prep.num_windows_in_part, prep.num_tokens_in_window)
windows_mat2 = make_windows_mat(prep.reordered_halves[1], prep.num_windows_in_part, prep.num_tokens_in_window)


def calc_y(w_mat, w_size, uniq):
    truncated_w_mat = w_mat[:, -w_size:]
    u = np.unique(truncated_w_mat, axis=0)
    num_total_windows = len(truncated_w_mat)
    num_uniq = len(u)
    num_repeated = num_total_windows - num_uniq
    #
    print(num_total_windows, num_uniq, num_repeated)
    if uniq:
        return len(u)
    else:
        return num_repeated


def plot(y_label, ys_list):
    bar_width0 = 0.0
    bar_width1 = 0.25
    _, ax = plt.subplots(dpi=192)
    ax.set_ylabel(y_label)
    ax.set_xlabel('punctuation')
    ax.set_xlabel('window_size')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    num_conditions = len(WINDOW_SIZES)
    xs = np.arange(1, num_conditions + 1)
    ax.set_xticks(xs)
    ax.set_xticklabels(WINDOW_SIZES)
    # plot
    colors = sns.color_palette("hls", 2)[::-1]
    labels = ['partition 1', 'partition 2']
    for n, (x, ys) in enumerate(zip(xs, ys_list)):
        ax.bar(x + bar_width0, ys[0], bar_width1, color=colors[0], label=labels[0] if n == 0 else '_nolegend_')
        ax.bar(x + bar_width1, ys[1], bar_width1, color=colors[1], label=labels[1] if n == 0 else '_nolegend_')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


plot('Number of repeated IO Mappings', [(calc_y(windows_mat1, ws, False), calc_y(windows_mat2, ws, False))
                                        for ws in WINDOW_SIZES])

plot('Number of unique IO Mappings', [(calc_y(windows_mat1, ws, True), calc_y(windows_mat2, ws, True))
                                      for ws in WINDOW_SIZES])