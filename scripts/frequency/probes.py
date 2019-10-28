import numpy as np
import matplotlib.pyplot as plt
import attr

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.utils import fit_line
from wordplay.utils import roll_mean

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-4096'

REVERSE = False
NUM_PARTS = 8
SHUFFLE_DOCS = False
NUM_MID_TEST_DOCS = 100

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=NUM_MID_TEST_DOCS,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)

# /////////////////////////////////////////////////////////////////

y = []
s = set(probe_store.vocab_ids)
for part in prep.reordered_parts:
    ones = [1 for w in part if w in s]
    num_occurrences = len(ones)
    y.append(num_occurrences)


# fig
fig, ax = plt.subplots(dpi=192, figsize=(6, 6))
plt.title('')
ax.set_ylabel('Num Probe Occurrences')
ax.set_xlabel('Partition')
ax.set_ylim([0, 40 * 1000])
# plot
x = np.arange(prep.num_parts)
ax.plot(x, y, '-', alpha=0.5)
y_fitted = fit_line(x, y)
ax.plot(x, y_fitted, '-')
y_rolled = roll_mean(y, 20)
ax.plot(x, y_rolled, '-')
fig.tight_layout()
plt.show()