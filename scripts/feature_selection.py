"""
What grammatical contexts (POS sequences) are most diagnostic of membership in the noun category,
 and how do they change with age?
"""
from typing import Tuple, Dict
from sortedcontainers import SortedSet, SortedDict
import attr
from collections import Counter
import numpy as np
from spacy.tokens.doc import Doc
from spacy.tokens.token import Token
from sklearn.feature_selection import mutual_info_classif

from aochildesnouns.targets import make_targets
from aochildesnouns.pre_processing import prepare_data
from aochildesnouns.params import Conditions


def get_target_contexts(targets: SortedSet,
                        doc: Doc,
                        direction: str,  # 'l' or 'r'
                        context_size: int,
                        preserve_order: bool,
                        min_num_contexts: int = 1,
                        verbose: bool = False,
                        ) -> Tuple[Dict[str, Tuple[str]], Dict[str, int]]:
    """
    collect POS tags of contexts in which targets occur.
    """

    target2contexts = SortedDict({t: [] for t in targets})
    context2f = {}
    for n, token in enumerate(doc[:-context_size]):  # iterate over spacy token objects

        token: Token

        if token.text not in targets:
            continue

        # get POS tag of context words
        if direction == 'l':
            context = tuple([t.tag_ for t in doc[n - context_size:n]])
        elif direction == 'r':
            context = tuple([t.tag_ for t in doc[n + 1:n + 1 + context_size]])
        else:
            raise AttributeError('Invalid arg to direction')

        if not context:
            continue

        # collect
        if preserve_order:
            target2contexts[token.text].append(context)
            try:
                context2f[context] += 1
            except KeyError:
                context2f[context] = 1
        else:
            single_word_contexts = [(w,) for w in context]
            target2contexts[token.text].extend(single_word_contexts)
            try:
                context2f[single_word_contexts] += 1
            except KeyError:
                context2f[single_word_contexts] = 1

    # exclude entries with too few contexts
    excluded = []
    included = SortedSet()
    for target, contexts in target2contexts.items():
        if len(contexts) < min_num_contexts:
            excluded.append(target)
        else:
            included.add(target)
    for target in excluded:
        if verbose:
            print(f'WARNING: Excluding "{target}" because it occurs {len(target2contexts[target])} times')
        del target2contexts[target]

    return target2contexts, SortedDict(context2f)


def make_target_vectors(target2contexts: Dict[str, Tuple[str]],
                        contexts_shared: SortedSet,
                        ) -> np.ndarray:
    """
    make target representations based on each target's contexts.
    representation can be BOW or preserve word-order, depending on how contexts were collected.
    """
    num_context_types = len(contexts_shared)

    assert '' not in target2contexts

    num_targets = len(target2contexts)
    context2col_id = {c: n for n, c in enumerate(contexts_shared)}

    res = np.zeros((num_targets, num_context_types), int)
    for row_id, (t, target_contexts) in enumerate(target2contexts.items()):

        assert target_contexts

        # make target representation
        c2f = Counter(target_contexts)
        for c, f in c2f.items():
            try:
                col_id = context2col_id[c]
            except KeyError:  # context is not in contexts_all (includes intersection of ctl and exp contexts)
                pass
            else:
                res[row_id, col_id] = int(f)

    # check each representation has information
    num_zero_rows = np.sum(~res.any(axis=1))
    if num_zero_rows > 0:
        print(f'WARNING: Found {num_zero_rows} targets with empty rows (targets that occur with non-shared contexts')
    # this assertion fails when only those context types are included that are shared by exp and ctl targets.
    # some targets only occur with those infrequent, non-shared contexts, and so do not have any counts in the matrix

    return res


for params in Conditions.all():

    age2doc = prepare_data(params)  # returns spacy Doc objects
    targets_exp, targets_ctl = make_targets(params, age2doc)

    # for each age
    for age, doc in sorted(age2doc.items(), key=lambda i: i[0]):

        params = attr.evolve(params,
                             age=age,
                             targets_control=True,
                             direction='l'
                             )
        print(params)

        # get experimental target representations
        target_exp2contexts, context_exp2f = get_target_contexts(targets_exp,
                                                                 doc,
                                                                 direction=params.direction,
                                                                 context_size=2,
                                                                 preserve_order=True,
                                                                 )

        # get control target representations
        target_ctl2contexts, context_ctl2f = get_target_contexts(targets_ctl,
                                                                 doc,
                                                                 direction=params.direction,
                                                                 context_size=2,
                                                                 preserve_order=True,
                                                                 )
        # make sure that the same contexts are in both representation matrices (an in the same order)
        contexts_shared = SortedSet(context_exp2f.copy().keys())
        contexts_shared.intersection_update(context_ctl2f.keys())

        # make representations
        target_exp_rep_mat = make_target_vectors(target_exp2contexts, contexts_shared)
        target_ctl_rep_mat = make_target_vectors(target_ctl2contexts, contexts_shared)
        print('shape of exp target representations={}'.format(target_exp_rep_mat.shape))
        print('shape of ctl target representations={}'.format(target_ctl_rep_mat.shape))

        # feature selection: which contexts are most diagnostic of category membership?
        X = np.vstack((target_exp_rep_mat, target_ctl_rep_mat))
        y = np.hstack((
            np.ones(len(target_exp_rep_mat), dtype=np.int) * 1,
            np.ones(len(target_ctl_rep_mat), dtype=np.int) * 0
        ))
        mutual_info = mutual_info_classif(X, y, discrete_features=True)  # mutual info for each column in X

        # print most diagnostic contexts
        for i in np.argsort(mutual_info)[::-1][:10]:
            c = contexts_shared[i]
            print(f'context={" ".join(c):<32} '
                  f'mutual-info={mutual_info[i]:.6f} '
                  f'f-exp={context_exp2f[c]:>6,} '
                  f'f-ctl={context_ctl2f[c]:>6,} ')