import numpy as np
import matplotlib.pyplot as plt
import attr
from scipy import stats

from preppy.legacy import TrainPrep

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.utils import split

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'  # _tags
PROBES_NAME = 'sem-4096'

NUM_PARTS = 32  # z-scoring doesn't make sense when num-parts=2
SHUFFLE_DOCS = False
START_MID = False
START_END = False

docs = load_docs(CORPUS_NAME,
                 shuffle_docs=SHUFFLE_DOCS,
                 start_at_midpoint=START_MID,
                 start_at_ends=START_END)

params = PrepParams(num_parts=NUM_PARTS)
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

MAX_F = 100000

# define rare words independently of partition
rare_words = set([w for w in prep.store.types if prep.store.w2f[w] < MAX_F])
num_rare = len(rare_words)
print(f'Found {num_rare} rare words')

print([w for w in prep.store.types if w not in rare_words])

# fig
fig, ax = plt.subplots(dpi=192)
plt.title(f'{num_rare} most infrequent words')
ax.set_xlabel('Partition')
ax.set_ylabel('z-scored Number of occurrences')
ax.yaxis.grid(False)
ax.set_ylim([-3, 3])
ax.axhline(y=0, color='grey')
# plot

# count the number of rare words in each partition
y = []
for tokens in split(prep.store.tokens, prep.num_tokens_in_part):
    num_rare_words = len([1 for w in tokens if w in rare_words])
    y.append(num_rare_words)

print(y)
ax.plot(stats.zscore(y), '-')
plt.show()






