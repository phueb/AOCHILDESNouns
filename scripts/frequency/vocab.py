import numpy as np
import matplotlib.pyplot as plt
import attr
from collections import Counter

from preppy import PartitionedPrep as TrainPrep
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

docs = load_docs(CORPUS_NAME)

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# /////////////////////////////////////////////////////////////////


dists = []
fig, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
for n, part in enumerate(prep.reordered_parts):
    c = Counter(part)
    dist = np.sort(np.log(list(c.values())))[::-1]
    dists.append(dist)
    ax.plot(dist, label=f'partition {n+1}')
ax.set_ylabel('Log Freq')
ax.set_xlabel('Word Id')
plt.legend()
plt.show()

