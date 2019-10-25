import random

from wordplay import config


def load_docs(corpus_name, shuffle_docs=False):
    with (config.Dirs.data / f'{corpus_name}.txt').open('r') as f:
        docs = f.readlines()
    num_docs = len(docs)
    print(f'Loaded {num_docs} documents from {corpus_name}')

    if shuffle_docs:
        print('Shuffling documents')
        random.shuffle(docs)

    return docs
