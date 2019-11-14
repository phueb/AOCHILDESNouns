from typing import List, Set


def split_into_sentences(tokens: List[str],
                         punctuation: Set[str],
                         ) -> List[List[str]]:
    assert isinstance(punctuation, set)

    res = [[]]
    for w in tokens:
        res[-1].append(w)
        if w in punctuation:
            res.append([])
    return res