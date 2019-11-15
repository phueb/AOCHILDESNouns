import numpy as np
import matplotlib.pyplot as plt
import attr

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay import config
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.utils import fit_line
from wordplay.utils import split
from wordplay.utils import roll_mean

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-all'

REVERSE = False
NUM_PARTS = 32
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

CATEGORY = 'tool'

y = []
s = set(probe_store.vocab_ids)
for tokens in split(prep.store.tokens, prep.num_tokens_in_part):
    ones = [w for w in tokens if w in probe_store.cat2probes[CATEGORY]]
    num_occurrences = len(ones)
    y.append(num_occurrences)
    print(num_occurrences)


# fig
fig, ax = plt.subplots(dpi=192, figsize=(6, 6))
plt.title(CATEGORY)
ax.set_ylabel(f'Number of occurrences')
ax.set_xlabel('Partition')
ax.set_ylim([0, 2 * 1000])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
x = np.arange(prep.num_parts)
ax.plot(x, y, '-', alpha=0.5)
y_fitted = fit_line(x, y)
ax.plot(x, y_fitted, '-')
fig.tight_layout()
plt.show()