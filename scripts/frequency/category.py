import numpy as np
import matplotlib.pyplot as plt
import attr

from preppy import PartitionedPrep as TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay import configs
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.io import load_docs
from wordplay.util import fit_line
from wordplay.util import split
from wordplay.util import roll_mean

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20191112_terms'
PROBES_NAME = 'sem-all'

REVERSE = False
NUM_PARTS = 32

NUM_MID_TEST_DOCS = 100

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=NUM_MID_TEST_DOCS,
                 num_test_take_random=0,
                 )

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

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
fig, ax = plt.subplots(dpi=configs.Fig.dpi, figsize=configs.Fig.fig_size)
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