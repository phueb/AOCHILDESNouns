from typing import List, Set


def get_sentences_from_tokens(tokens: List[str],
                              punctuation: Set[str],
                              ) -> List[List[str]]:
    assert isinstance(punctuation, set)

    res = [[]]
    for w in tokens:
        res[-1].append(w)
        if w in punctuation:
            res.append([])
    return res