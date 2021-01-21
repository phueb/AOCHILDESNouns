import numpy as np
import matplotlib.pyplot as plt
import attr
from scipy import stats

from preppy import PartitionedPrep as TrainPrep

from wordplay import configs
from wordplay.params import PrepParams
from wordplay.io import load_docs
from wordplay.pos import pos2tags
from wordplay.util import fit_line
from wordplay.util import roll_mean
from wordplay.util import split

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319_tags'  # must have spacy tags
PROBES_NAME = 'sem-all'

REVERSE = False
NUM_PARTS = 32  # z-score does not make sense with num_parts=2
docs = load_docs(CORPUS_NAME)

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE)
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

POS_LIST = ['NOUN']

for pos in POS_LIST or sorted(pos2tags.keys()):

    pos_tags = set(pos2tags[pos])

    y = []
    for tags in split(prep.store.tokens, prep.num_tokens_in_part):
        ones = [1 for tag in tags if tag in pos_tags]
        num = len(ones)
        y.append(num)

    # fig
    _, ax = plt.subplots(figsize=configs.Fig.fig_size, dpi=configs.Fig.dpi)
    plt.title('')
    ax.set_ylabel(f'Num {pos}s')
    ax.set_xlabel('Partition')
    ax.set_ylim([0, 100 * 1000])
    # plot
    x = np.arange(params.num_parts)
    ax.plot(x, y, '-', alpha=0.5)
    y_fitted = fit_line(x, y)
    ax.plot(x, y_fitted, '-')
    y_rolled = roll_mean(y, 20)
    ax.plot(x, y_rolled, '-')
    plt.show()

    # fig
    _, ax = plt.subplots(figsize=configs.Fig.fig_size, dpi=configs.Fig.dpi)
    plt.title('')
    ax.set_ylabel(f'Z-scored Num {pos}s')
    ax.set_xlabel('Partition')
    # plot
    ax.axhline(y=0, color='grey')
    x = np.arange(params.num_parts)
    ax.plot(x, stats.zscore(y), '-', alpha=1.0)
    plt.show()