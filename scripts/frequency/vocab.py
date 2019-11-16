import numpy as np
import matplotlib.pyplot as plt
import attr
from collections import Counter

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay import config
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'  # _tags
PROBES_NAME = 'sem-all'

REVERSE = False
NUM_PARTS = 8
SHUFFLE_DOCS = False
START_MID = False
START_END = False

docs = load_docs(CORPUS_NAME,
                 shuffle_docs=SHUFFLE_DOCS,
                 start_at_midpoint=START_MID,
                 start_at_ends=START_END)

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# /////////////////////////////////////////////////////////////////


dists = []
fig, ax = plt.subplots(dpi=config.Fig.dpi)
for n, part in enumerate(prep.reordered_parts):
    c = Counter(part)
    dist = np.sort(np.log(list(c.values())))[::-1]
    dists.append(dist)
    ax.plot(dist, label=f'partition {n+1}')
ax.set_ylabel('Log Freq')
ax.set_xlabel('Word Id')
plt.legend()
plt.show()

