import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import attr

from preppy import PartitionedPrep as TrainPrep

from wordplay import config
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.measures import calc_entropy
from wordplay.measures import mtld

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-all'


NUM_MID_TEST_DOCS = 0
NUM_PARTS = 64

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=NUM_MID_TEST_DOCS,
                 num_test_take_random=0,
                 )

params = PrepParams(num_parts=NUM_PARTS)
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

AX_FONTSIZE = 8
LEG_FONTSIZE = 6
FIGSIZE = (3.2, 2.2)
DPI = config.Fig.dpi
IS_LOG = True
WSPACE = 0.0
HSPACE = 0.0
WPAD = 0.0
HPAD = 0.0
PAD = 0.2
LW = 0.5

# xys
ys = [
    [calc_entropy(part) for part in prep.reordered_parts],
    [mtld(part) for part in prep.reordered_parts]
]

# fig
y_labels = ['Shannon Entropy', 'MTLD']
fig, axs = plt.subplots(2, 1, dpi=config.Fig.dpi, figsize=config.Fig.fig_size)
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
