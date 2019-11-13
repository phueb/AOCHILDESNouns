"""
Research questions:
1. How frequent are early noun contexts in partition 1 and 2 of AO-CHILDES?

"""

import numpy as np
import matplotlib.pyplot as plt
import attr

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.utils import get_sliding_windows
from wordplay.utils import split
from wordplay.contexts import make_word2contexts

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'syn-4096'

REVERSE = False
NUM_PARTS = 32
SHUFFLE_DOCS = False

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=0,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)

# /////////////////////////////////////////////////////////////////

CONTEXT_SIZE = 3
POS = 'NOUN'

# pos_words
pos_words = probe_store.cat2probes[POS]

# get all contexts for pos_words
w2contexts = make_word2contexts(prep.store.tokens, pos_words, CONTEXT_SIZE)
unique_contexts = set()
for contexts in w2contexts.values():
    for c in contexts:
        unique_contexts.add(c)
num_contexts = len(unique_contexts)
print(f'Found {num_contexts} {POS} contexts with size={CONTEXT_SIZE}')

# get locations of contexts
context2locations = {c: [] for n, c in enumerate(unique_contexts)}
contexts_in_order = [c for c in get_sliding_windows(CONTEXT_SIZE, prep.store.tokens)
                     if c in unique_contexts]
for loc, c in enumerate(contexts_in_order):
    context2locations[c].append(loc)

# split contexts into early and late
print('Splitting contexts into early and late')
sorted_contexts = sorted(unique_contexts, key=lambda c: np.mean(context2locations[c]))
contexts1 = set(sorted_contexts[:num_contexts // 2])   # earliest
contexts2 = set(sorted_contexts[-num_contexts // 2:])  # latest

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
fig, ax = plt.subplots(dpi=192)
plt.title(f'context-size={CONTEXT_SIZE}')
x = np.arange(NUM_PARTS)
ax.set_xlabel('Partition')
ax.set_ylabel(f'{POS} Context Token Frequency')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.yaxis.grid(True, alpha=0.1)
# plot
ax.plot(y1, label='early contexts')
ax.plot(y2, label='late contexts')

ax.legend(frameon=False)
fig.tight_layout()
plt.show()
