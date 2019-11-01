"""
Research questions:
1. does partition 1 have fewer SVO triples? in other words, are fewer meanings expressed?
"""


import numpy as np
import matplotlib.pyplot as plt
import attr
from scipy import stats
import spacy
from spacy.tokens import Span
import pyprind

from preppy.legacy import TrainPrep

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay import config
from wordplay.utils import fit_line
from wordplay.utils import split
from wordplay.sentences import get_sentences_from_tokens

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-4096'

REVERSE = False
NUM_PARTS = 32  # z-score does not make sense with num_parts=2
SHUFFLE_DOCS = False

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=0,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE)
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

VERBOSE = False

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm", disable=['ner'])


def contains_symbol(span):
    """
    checks if span has any undesired symbols.
    used to filter noun chunks.
    """
    return any(s in span.text for s in set(config.Symbols.all))


Span.set_extension("contains_symbol", getter=contains_symbol)


y = []
pbar = pyprind.ProgBar(NUM_PARTS, stream=2) if not VERBOSE else None
for tokens in split(prep.store.tokens, prep.num_tokens_in_part):
    sentences = get_sentences_from_tokens(tokens, punctuation={'.', '!', '?'})
    texts = [' '.join(s) for s in sentences]

    noun_chunks_in_part = []
    for doc in nlp.pipe(texts):
        for chunk in doc.noun_chunks:
            if not chunk._.contains_symbol:
                noun_chunks_in_part.append(chunk.text)

    num_chunks_in_part = len(noun_chunks_in_part)
    num_unique_chunks_in_part = len(set(noun_chunks_in_part))
    if VERBOSE:
        print(f'Found {num_chunks_in_part:>12,} noun chunks')
        print(f'Found {num_unique_chunks_in_part:>12,} unique noun chunks')
    else:
        pbar.update()

    y.append(num_unique_chunks_in_part)


# fig
_, ax = plt.subplots(dpi=192)
plt.title('Noun chunks')
ax.set_ylabel('Num unique noun chunks')
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
_, ax = plt.subplots(dpi=192)
plt.title('Noun chunks')
ax.set_ylabel(f'Z-scored Num unique noun chunks')
ax.set_xlabel('Partition')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
ax.axhline(y=0, color='grey', linestyle=':')
x = np.arange(params.num_parts)
ax.plot(x, stats.zscore(y), alpha=1.0)
plt.show()