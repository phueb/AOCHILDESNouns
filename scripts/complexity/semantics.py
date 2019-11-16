"""
Research questions:
1. does partition 1 have fewer unique noun chunks?
"""


import numpy as np
import matplotlib.pyplot as plt
import attr
from scipy import stats
import spacy
import pyprind

from preppy.legacy import TrainPrep

from wordplay import config
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.utils import fit_line
from wordplay.utils import split
from wordplay.sentences import split_into_sentences
from wordplay.svo import subject_verb_object_triples

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-all'

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

VERBOSE = True

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm", disable=['ner'])


y = []
pbar = pyprind.ProgBar(NUM_PARTS, stream=2) if not VERBOSE else None
for tokens in split(prep.store.tokens, prep.num_tokens_in_part):
    sentences = split_into_sentences(tokens, punctuation={'.', '!', '?'})
    texts = [' '.join(s) for s in sentences]

    triples_in_part = []
    for doc in nlp.pipe(texts):
        triples = [t for t in subject_verb_object_triples(doc)]  # only returns triples, not partial triples
        triples_in_part += triples
    num_triples_in_part = len(triples_in_part)
    num_unique_triples_in_part = len(set(triples_in_part))

    if VERBOSE:
        print(f'Found {num_triples_in_part:>12,} SVO triples')
        print(f'Found {num_unique_triples_in_part:>12,} unique SVO triples')
    else:
        pbar.update()

    y.append(num_unique_triples_in_part)


# fig
_, ax = plt.subplots(dpi=config.Fig.dpi)
plt.title('SVO-triples')
ax.set_ylabel('Num unique SVO-triples')
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
_, ax = plt.subplots(dpi=config.Fig.dpi)
plt.title('SVO-triples')
ax.set_ylabel(f'Z-scored Num unique SVO-triples')
ax.set_xlabel('Partition')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
ax.axhline(y=0, color='grey', linestyle=':')
x = np.arange(params.num_parts)
ax.plot(x, stats.zscore(y), alpha=1.0)
plt.show()