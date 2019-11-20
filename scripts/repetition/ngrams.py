"""
Research question:
1. Are n-grams in partition 1 of AO-CHILDES repeated more often on average compared to partition 2?
"""

import attr
from tabulate import tabulate

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.utils import get_sliding_windows

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-all'

docs = load_docs(CORPUS_NAME)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# /////////////////////////////////////////////////////////////////


NGRAM_SIZES = [1, 2, 3, 4, 5, 6, 7]

tokens1 = prep.store.tokens[:prep.midpoint]
tokens2 = prep.store.tokens[-prep.midpoint:]

rows = []
for ngram_size in NGRAM_SIZES:

    ngrams1 = get_sliding_windows(ngram_size, tokens1)
    ngrams2 = get_sliding_windows(ngram_size, tokens2)
    num_ngrams1 = len(ngrams1)
    num_ngrams2 = len(ngrams2)

    unique_ngrams1 = set(ngrams1)
    unique_ngrams2 = set(ngrams2)
    ngram_set_len1 = len(unique_ngrams1)
    ngram_set_len2 = len(unique_ngrams2)

    y1 = (ngram_set_len1 + num_ngrams1) / ngram_set_len1
    y2 = (ngram_set_len2 + num_ngrams2) / ngram_set_len2
    rows.append((ngram_size, y1, y2))

# print table
headers = ['N-gram size', 'partition 1', 'partition 2']
print(tabulate(rows,
               headers=headers,
               tablefmt='simple',
               floatfmt='.2f'))