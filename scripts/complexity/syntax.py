"""
Research questions:
1. Is partition 1 syntactically more complex? Does it have more unique POS-tag sequences (must be sentences)?
"""


import numpy as np
import matplotlib.pyplot as plt
import attr
from scipy import stats

from preppy import PartitionedPrep as TrainPrep

from wordplay import configs
from wordplay.params import PrepParams
from wordplay.io import load_docs
from wordplay.util import fit_line, split_into_sentences
from wordplay.util import split

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319_tags'  # must have spacy tags
PROBES_NAME = 'sem-all'

REVERSE = False
NUM_PARTS = 8  # z-score does not make sense with num_parts=2

NUM_MID_TEST_DOCS = 100

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=NUM_MID_TEST_DOCS,
                 num_test_take_random=0,
                 )

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE)
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

y = []
for tags in split(prep.store.tokens, prep.num_tokens_in_part):

    sentences = split_into_sentences(tags, punctuation={'.'})
    unique_sentences = np.unique(sentences)
    print(f'Found {len(sentences):>12,} total sentences in part')
    print(f'Found {len(unique_sentences):>12,} unique sentences in part')

    y.append(len(unique_sentences))


# fig
_, ax = plt.subplots(figsize=configs.Fig.fig_size, dpi=configs.Fig.dpi)
plt.title('')
ax.set_ylabel('Num unique tag-sequences')
ax.set_xlabel('Partition')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
x = np.arange(params.num_parts)
ax.plot(x, y, '-', alpha=0.5)
y_fitted = fit_line(x, y)
ax.plot(x, y_fitted, '-')
plt.show()

# fig
_, ax = plt.subplots(figsize=configs.Fig.fig_size, dpi=configs.Fig.dpi)
plt.title('Syntactic Complexity')
ax.set_ylabel(f'Z-scored Num unique tag-sequences')
ax.set_xlabel('Partition')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
ax.axhline(y=0, color='grey', linestyle=':')
x = np.arange(params.num_parts)
ax.plot(x, stats.zscore(y), alpha=1.0)
plt.show()