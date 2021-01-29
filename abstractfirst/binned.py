import numpy as np
from itertools import groupby
from typing import List, Dict

from abstractfirst import configs
from abstractfirst.util import split


def make_age_bin2data(corpus_name: str,
                      age_step: int,
                      verbose: bool = False,
                      ) -> Dict[float, List[str]]:

    # convert ages to age bins
    ages_path = configs.Dirs.corpora / f'{corpus_name}_ages.txt'
    ages_text = ages_path.read_text(encoding='utf-8')
    ages = np.array(ages_text.split(), dtype=np.float)
    ages_binned = ages - np.mod(ages, age_step)
    ages_binned = ages_binned.astype(np.int)

    # load data
    data_path = configs.Dirs.corpora / f'{corpus_name}.txt'
    data_text = data_path.read_text(encoding='utf-8')
    data_by_doc = [doc.split() for doc in data_text.split('\n')[:-1]]

    age_and_docs = zip(ages_binned, data_by_doc)

    res = {}
    for age_bin, doc_group in groupby(age_and_docs, lambda d: d[0]):
        docs = [d[1] for d in doc_group]
        data = list(np.concatenate(docs))
        if verbose:
            print(f'Found {len(docs)} transcripts for age-bin={age_bin}')

        res[age_bin] = data

    return res


def adjust_binned_data(age_bin2data: Dict[float, List[str]],
                       min_num_data: int,
                       no_binning: bool = False,
                       ):
    """
    return dictionary similar to input but with a constant number of data per age_bin.
    combine bins when a bin is too small.
    """

    if no_binning:
        print('WARNING: Not binning by age')
        all_tokens = np.concatenate(list(age_bin2data.values()))
        return {n: list(tokens) for n, tokens in enumerate(split(all_tokens, split_size=min_num_data))
                if len(tokens) == min_num_data}

    res = {}
    buffer = []
    for age_bin, data in age_bin2data.items():

        buffer += data

        if len(buffer) > min_num_data:
            res[age_bin] = buffer[-min_num_data:]
            buffer = []
        else:
            continue

    return res