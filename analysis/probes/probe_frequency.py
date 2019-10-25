import numpy as np
import matplotlib.pyplot as plt
import attr

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.utils import fit_line
from wordplay.utils import roll_mean

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-4096'

REVERSE = False
NUM_PARTS = 256
SHUFFLE_DOCS = False

DPI = 192
FIG_SIZE = (6, 6)

docs = load_docs(CORPUS_NAME, SHUFFLE_DOCS)

params = attr.asdict(PrepParams(num_parts=NUM_PARTS))
prep = TrainPrep(docs, **params)


probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)
y = []
for part in prep.reordered_parts:
    intersection = set(part) & set(probe_store.vocab_ids)
    num_occurrences = len(intersection)
    y.append(num_occurrences)


# fig
fig, ax = plt.subplots(dpi=DPI, figsize=FIG_SIZE)
plt.title('')
ax.set_ylabel('Num Probe Occurrences')
ax.set_xlabel('Partition')
# plot
x = np.arange(prep.num_parts)
ax.plot(x, y, '-', alpha=0.5)
y_fitted = fit_line(x, y)
ax.plot(x, y_fitted, '-')
y_rolled = roll_mean(y, 20)
ax.plot(x, y_rolled, '-')
fig.tight_layout()
plt.show()