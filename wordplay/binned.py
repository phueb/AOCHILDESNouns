import numpy as np
from itertools import groupby
from typing import List, Dict

from wordplay import config


"""
Note:
age information is not available for childes-20180319
"""


def make_age_bin2tokens(corpus_name: str,
                        age_step: int,
                        suffix: str = '_terms',
                        verbose: bool = False,
                        ) -> Dict[float, List[str]]:

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

    age_bin2tokens = {}
    for age_bin, data_group in groupby(data, lambda d: d[0]):
        docs = [d[1] for d in data_group]
        tokens = list(np.concatenate(docs))
        if verbose:
            print(f'Found {len(docs)} transcripts for age-bin={age_bin}')

        age_bin2tokens[age_bin] = tokens

    return age_bin2tokens