import numpy as np
from itertools import groupby

from wordplay import config


"""
Note:
age information is not available for childes-20180319
"""


def get_binned(corpus_name: str,
               age_step: int,
               suffix: str = '_terms',
               ) -> tuple:

    ages_path = config.Dirs.corpora / f'{corpus_name}_ages.txt'
    ages_text = ages_path.read_text(encoding='utf-8')
    ages = np.array(ages_text.split(), dtype=np.float)
    tags_path = config.Dirs.corpora / f'{corpus_name}{suffix}.txt'
    tags_text = tags_path.read_text(encoding='utf-8')
    tags_by_doc = [doc.split() for doc in tags_text.split('\n')[:-1]]
    ages_binned = ages - np.mod(ages, age_step)

    # convert ages to age bins
    ages_binned = ages_binned.astype(np.int)
    data = zip(ages_binned, tags_by_doc)

    tokens_by_binned_age = []
    labels = []
    for binned_age, data_group in groupby(data, lambda d: d[0]):
        docs = [d[1] for d in data_group]
        tags = list(np.concatenate(docs))
        print(f'Found {len(docs)} transcripts for age-bin={binned_age}')

        tokens_by_binned_age.append(tags)
        labels.append(str(binned_age))

    return labels, tokens_by_binned_age