import attr
from tabulate import tabulate

from preppy import PartitionedPrep as TrainPrep
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


NGRAM_SIZES = [1, 2, 3]

tokens1 = prep.store.tokens[:prep.midpoint]
tokens2 = prep.store.tokens[-prep.midpoint:]

rows = []
for ngram_size in NGRAM_SIZES:
    probes_in_tokens1 = [token for token in tokens1 if token in probe_store.types]
    probes_in_tokens2 = [token for token in tokens2 if token in probe_store.types]
    num_total_probes1 = len(probes_in_tokens1)
    num_total_probes2 = len(probes_in_tokens2)
    rows.append((f'Probes ({PROBES_NAME})', num_total_probes1, num_total_probes2))

    ngrams1 = get_sliding_windows(ngram_size, tokens1)
    ngrams2 = get_sliding_windows(ngram_size, tokens2)
    num_ngrams1 = len(ngrams1)
    num_ngrams2 = len(ngrams2)
    rows.append((f'{ngram_size}-gram Tokens', num_ngrams1, num_ngrams2))

    unique_ngrams1 = set(ngrams1)
    unique_ngrams2 = set(ngrams2)
    ngram_set_len1 = len(unique_ngrams1)
    ngram_set_len2 = len(unique_ngrams2)
    rows.append((f'{ngram_size}-gram Types', ngram_set_len1, ngram_set_len2))

    print('num n-grams in 1 also in 2:')
    print(len([ngram for ngram in unique_ngrams1 if ngram in unique_ngrams2]))
    print()

    unique_probe_ngrams1 = set([ngram for ngram in ngrams1 if any([t in probe_store.types for t in ngram])])
    unique_probe_ngrams2 = set([ngram for ngram in ngrams2 if any([t in probe_store.types for t in ngram])])
    ngram_probes_set_len1 = len(unique_probe_ngrams1)
    ngram_probes_set_len2 = len(unique_probe_ngrams2)
    rows.append((f'{ngram_size}-gram tokens with probes', ngram_probes_set_len1, ngram_probes_set_len2))


# print table
headers = ['Count Type', 'partition 1', 'partition 2']
print(tabulate(rows,
               headers=headers,
               tablefmt='simple'))