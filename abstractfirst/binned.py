import numpy as np
from itertools import groupby
from typing import List, Dict, Tuple

from abstractfirst import configs
from abstractfirst.params import Params


def make_age_bin2data(params: Params,
                      verbose: bool = False,
                      ) -> Dict[float, str]:

    # convert ages to age bins
    ages_path = configs.Dirs.corpora / f'{params.corpus_name}_ages.txt'
    ages_text = ages_path.read_text(encoding='utf-8')
    ages = np.array(ages_text.split(), dtype=np.float)
    ages_binned = ages - np.mod(ages, params.age_step)
    ages_binned = ages_binned.astype(np.int)

    # load data
    data_path = configs.Dirs.corpora / f'{params.corpus_name}.txt'
    data_text = data_path.read_text(encoding='utf-8')
    data_by_doc: List[str] = [doc for doc in data_text.split('\n')[:-1]]

    age_and_docs = zip(ages_binned, data_by_doc)

    res = {}
    for age_bin, age_and_doc_group in groupby(age_and_docs, lambda d: d[0]):
        age_and_doc_group: List[Tuple[int, str]]
        num_docs = 0
        data: str = ''
        for _, doc in age_and_doc_group:
            data += doc
            num_docs += 1

        if verbose:
            print(f'Found {num_docs} transcripts for age-bin={age_bin}')

        res[age_bin] = data

    return res


def adjust_binned_data(age_bin2text: Dict[float, str],
                       min_num_tokens: int,
                       ) -> Dict[float, str]:
    """
    return dictionary similar to input but with a constant number of tokens per age_bin.
    combine bins when a bin is too small.
    """

    res = {}
    token_buffer = []
    for age_bin, text in age_bin2text.items():

        tokens_in_text = text.split()
        token_buffer.extend(tokens_in_text)

        num_tokens_in_buffer = len(token_buffer)
        if num_tokens_in_buffer > min_num_tokens:
            res[age_bin] = ' '.join(token_buffer[-min_num_tokens:])
            token_buffer.clear()
        else:
            continue

    return res
