import attr

from preppy import PartitionedPrep

from wordplay.pos import make_pos_words
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay import config

CORPUS_NAME = 'childes-20191206'

docs = load_docs(CORPUS_NAME)
params = PrepParams(num_types=None)
prep = PartitionedPrep(docs, **attr.asdict(params))

nouns = make_pos_words(prep.store.types, 'NOUN')
verbs = make_pos_words(prep.store.types, 'VERB')
adjectives = make_pos_words(prep.store.types, 'ADJ')

nouns_path = config.Dirs.words / f'{CORPUS_NAME}-nouns-{len(nouns)}.txt'
with nouns_path.open('w') as f:
    for w in nouns:
        f.write(w + '\n')

verbs_path = config.Dirs.words / f'{CORPUS_NAME}-verbs-{len(verbs)}.txt'
with verbs_path.open('w') as f:
    for w in verbs:
        f.write(w + '\n')

adjectives_path = config.Dirs.words / f'{CORPUS_NAME}-adjs-{len(adjectives)}.txt'
with adjectives_path.open('w') as f:
    for w in adjectives:
        f.write(w + '\n')