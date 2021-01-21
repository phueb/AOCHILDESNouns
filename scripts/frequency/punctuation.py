import attr
from functools import reduce
from operator import iconcat

from preppy import PartitionedPrep as TrainPrep

from wordplay.params import PrepParams
from wordplay.io import load_docs

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20191206'
docs = load_docs(CORPUS_NAME, num_test_take_random=0)
params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

num_p = prep.store.tokens.count('.')
num_e = prep.store.tokens.count('!')
num_q = prep.store.tokens.count('?')
num_punctuation = num_p + num_q + num_e
print(f'{num_punctuation:,}')

tokenized_docs = [d.split() for d in docs]
tokens = reduce(iconcat, tokenized_docs, [])

num_p = tokens.count('.')
num_e = tokens.count('!')
num_q = tokens.count('?')
num_punctuation = num_p + num_q + num_e
print(f'{num_punctuation:,}')