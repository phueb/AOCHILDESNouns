import numpy as np
import matplotlib.pyplot as plt
import attr

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.pos import pos2tags
from wordplay.utils import fit_line
from wordplay.utils import roll_mean
from wordplay.utils import split

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319_tags'
PROBES_NAME = 'sem-4096'

REVERSE = False
NUM_PARTS = 8
SHUFFLE_DOCS = True
START_MID = False
START_END = False

docs = load_docs(CORPUS_NAME,
                 shuffle_docs=SHUFFLE_DOCS,
                 start_at_midpoint=START_MID,
                 start_at_ends=START_END)

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)

# /////////////////////////////////////////////////////////////////

POS_LIST = ['noun']

for pos in POS_LIST or sorted(pos2tags.keys()):

    pos_tags = set(pos2tags[pos])

    y = []
    for tags in split(prep.store.tokens, prep.num_tokens_in_part):
        ones = [1 for tag in tags if tag in pos_tags]
        num = len(ones)
        y.append(num)

    # fig
    _, ax = plt.subplots(dpi=192)
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