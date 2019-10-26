import random
from typing import List

from wordplay import config


childes_mid_doc_ids = tuple(range(1500, 1600))


def load_docs(corpus_name: str,
              shuffle_docs: bool = False,
              test_from_middle: bool = False,
              num_test_docs: int = 100,
              start_at_midpoint: bool = False,
              start_at_ends: bool = False,
              split_seed: int = 3,
              shuffle_seed: int = 20,  # 20 results in pretty even probe distribution
              ) -> List[str]:

    p = config.Dirs.corpora / f'{corpus_name}.txt'
    docs = p.read_text().split('\n')
    num_docs = len(docs)
    print(f'Loaded {num_docs} documents from {corpus_name}')

    assert not(shuffle_docs and start_at_midpoint and start_at_ends)
    assert not(shuffle_docs and start_at_midpoint)
    assert not(shuffle_docs and start_at_ends)
    assert not(start_at_midpoint and start_at_ends)

    if test_from_middle:
        test_doc_ids = childes_mid_doc_ids
    else:
        num_test_doc_ids = num_docs - num_test_docs
        random.seed(split_seed)
        test_doc_ids = random.sample(range(num_test_doc_ids), num_test_docs)

    # split train/test
    print('Splitting docs into train and test...')
    test_docs = []
    for test_line_id in test_doc_ids:
        test_doc = docs.pop(test_line_id)  # removes line and returns removed line
        test_docs.append(test_doc)

    if shuffle_docs:
        random.seed(shuffle_seed)
        print('Shuffling documents')
        random.shuffle(docs)

    if start_at_midpoint:
        docs = reorder_docs_from_midpoint(docs)

    if start_at_ends:
        docs = reorder_docs_from_ends(docs)

    return docs


def reorder_docs_from_midpoint(docs: List[str]
                               ) -> List[str]:
    """
    reorder docs such that first docs are docs that are most central
    """
    # start, middle, end
    s = 0
    e = len(docs)
    m = e // 2

    res = []
    for i, j in zip(range(m, e + 1)[::+1],
                    range(s, m + 0)[::-1]):
        res += [docs[i], docs[j]]

    assert len(res) == len(docs)

    return res


def reorder_docs_from_ends(docs: List[str]
                           ) -> List[str]:
    """
    reorder docs such that first docs are docs that are from ends
    """
    # start, middle, end
    s = 0
    e = len(docs)
    m = e // 2

    res = []
    for i, j in zip(range(m, e + 0)[::-1],
                    range(s, m + 1)[::+1]):
        res += [docs[i], docs[j]]

    assert len(res) == len(docs)

    return res
