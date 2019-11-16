import numpy as np
from itertools import groupby
from typing import List, Dict

from wordplay import config
from wordplay.word_sets import excluded
from wordplay.utils import split


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


def make_age_bin2tokens_with_min_size(age_bin2tokens: Dict[float, List[str]],
                                      min_num_tokens: int,
                                      no_binning: bool,
                                      ):
    """
    return dictionary similar to age_bin2tokens but with a constant number of tokens per age_bin.
    combine bins when a bin is too small.
    remove content from bin when a bin is too small
    """

    if no_binning:
        print('WARNING: Not binning by age')
        all_tokens = np.concatenate(list(age_bin2tokens.values()))
        return {n: list(tokens) for n, tokens in enumerate(split(all_tokens, split_size=min_num_tokens))
                if len(tokens) == min_num_tokens}

    res = {}
    buffer = []
    for age_bin, tokens in age_bin2tokens.items():

        buffer += tokens

        if len(buffer) > min_num_tokens:
            res[age_bin] = buffer[-min_num_tokens:]
            buffer = []
        else:
            continue

    return res