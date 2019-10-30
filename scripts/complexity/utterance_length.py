import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import attr

from preppy.legacy import TrainPrep

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.stats import calc_utterance_lengths

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-4096'

SHUFFLE_DOCS = False
NUM_MID_TEST_DOCS = 0

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=NUM_MID_TEST_DOCS,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams(num_parts=2)
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

AX_FONTSIZE = 8
LEG_FONTSIZE = 6
FIGSIZE = (3.2, 2.2)
DPI = 192
IS_LOG = True
WSPACE = 0.0
HSPACE = 0.0
WPAD = 0.0
HPAD = 0.0
PAD = 0.2
LW = 0.5

# xys
ys = [calc_utterance_lengths(prep.store.tokens, is_avg=True),
      calc_utterance_lengths(prep.store.tokens, is_avg=False)]

# fig
y_labels = ['Mean Utterance\nLength', 'Std Utterance\nLength']
fig, axs = plt.subplots(2, 1, dpi=DPI, figsize=FIGSIZE)
for ax, y_label, y in zip(axs, y_labels, ys):
    if ax == axs[-1]:
        ax.set_xlabel('Corpus Location', fontsize=AX_FONTSIZE, labelpad=-10)
        ax.set_xticks([0, len(y)])
        ax.set_xticklabels(['0', f'{prep.store.num_tokens:,}'])
        plt.setp(ax.get_xticklabels(), fontsize=AX_FONTSIZE)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylabel(y_label, fontsize=LEG_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    plt.setp(ax.get_yticklabels(), fontsize=LEG_FONTSIZE)
    # plot
    ax.plot(y, linewidth=LW, label=y_label, c='black')
# show
plt.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
plt.tight_layout(h_pad=HPAD, w_pad=WPAD, pad=PAD)
plt.show()
