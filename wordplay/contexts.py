from typing import Set, List, Dict


def make_word2contexts(tokens: List[str],
                       words: Set[str],
                       context_size: int,
                       ) -> Dict[str, List[int]]:

    print('Making w2contexts...')
    res = {w: [] for w in words}
    for loc, w in enumerate(tokens):
        if w in words:
            context = tuple(w for w in tokens[loc - context_size:loc])
            res[w].append(context)
    return res