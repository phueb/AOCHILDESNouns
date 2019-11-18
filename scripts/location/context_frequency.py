"""
Research questions:
1. How frequent are early noun contexts in partition 1 and 2 of AO-CHILDES?

"""

import numpy as np
import matplotlib.pyplot as plt
import attr
import math

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay import config
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.utils import get_sliding_windows
from wordplay.utils import split
from wordplay.contexts import make_word2contexts

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'syn-4096'

REVERSE = False
NUM_PARTS = 8


docs = load_docs(CORPUS_NAME)

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# /////////////////////////////////////////////////////////////////

CONTEXT_SIZE = 2
POS = 'VERB'
NUM_CONTEXTS = 10 * 1000  # number of first and last unique contexts
MEDIAN_SPLIT = False

# pos_words
pos_words = probe_store.cat2probes[POS]

# get all contexts for pos_words
w2contexts = make_word2contexts(prep.store.tokens, pos_words, CONTEXT_SIZE)
unique_contexts = set()
for contexts in w2contexts.values():
    for c in contexts:
        unique_contexts.add(c)
num_contexts = len(unique_contexts)
print(list(unique_contexts)[:20])
print(f'Found {num_contexts} {POS} contexts with size={CONTEXT_SIZE}')

# get locations of contexts
context2locations = {c: [] for n, c in enumerate(unique_contexts)}
contexts_in_order = [c for c in get_sliding_windows(CONTEXT_SIZE, prep.store.tokens)
                     if c in unique_contexts]
for loc, c in enumerate(contexts_in_order):
    context2locations[c].append(loc)

# split contexts into early and late
print('Splitting contexts into early and late')
if MEDIAN_SPLIT:
    sorted_contexts = sorted(unique_contexts, key=lambda c: np.mean(context2locations[c]))
    contexts1 = set(sorted_contexts[:num_contexts // 2])   # earliest
    contexts2 = set(sorted_contexts[-num_contexts // 2:])  # latest
    title = f'Median-split of contexts\ncontext-size={CONTEXT_SIZE}'
else:
    # get first N and last N unique contexts
    contexts1 = set(list(dict.fromkeys(contexts_in_order))[:NUM_CONTEXTS])
    contexts2 = set(list(dict.fromkeys(contexts_in_order[::-1]))[:NUM_CONTEXTS])
    title = f'First and last {NUM_CONTEXTS} contexts\ncontext-size={CONTEXT_SIZE}'

assert len(contexts1) == len(contexts2)

# count early and late contexts in parts
y1 = []
y2 = []
for part_id, tokens in enumerate(split(prep.store.tokens, prep.num_tokens_in_part)):
    print(f'Counting contexts in part={part_id}')

    contexts_in_tokens = get_sliding_windows(CONTEXT_SIZE, tokens)
    y1i = len([c for c in contexts_in_tokens if c in contexts1])
    y2i = len([c for c in contexts_in_tokens if c in contexts2])

    y1.append(y1i)
    y2.append(y2i)

# fig
fig, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
plt.title(title)
x = np.arange(NUM_PARTS)
ax.set_xlabel('Partition')
ax.set_ylabel(f'{POS} Context Token Frequency')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.yaxis.grid(True, alpha=0.1)
plt.grid(True, which='both', axis='y', alpha=0.2)
max_y = np.max(np.concatenate((y1, y2)))
ax.set_ylim([0, math.ceil(max_y)])
# plot
ax.plot(y1, label='first contexts')
ax.plot(y2, label='last contexts')

ax.legend(frameon=False, loc='lower left')
fig.tight_layout()
plt.show()
