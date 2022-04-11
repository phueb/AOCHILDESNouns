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
from sklearn.feature_selection import chi2

from aochildesnouns.targets import make_targets
from aochildesnouns.pre_processing import prepare_data
from aochildesnouns.params import Conditions

CONTEXT_SIZE = 1


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
            raise AttributeError(f'Invalid arg to direction ({direction})')

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
                        max_sum: int,  # maximum number of co-occurrence observations
                        ) -> np.ndarray:
    """
    make target representations based on each target's contexts.
    representation can be BOW or preserve word-order, depending on how contexts were collected.
    """
    num_context_types = len(contexts_shared)

    assert '' not in target2contexts

    num_targets = len(target2contexts)
    context2col_id = {c: n for n, c in enumerate(contexts_shared)}

    res = np.zeros((num_targets, num_context_types), float) + 0.0001
    collected_f = 0
    total_f = 0
    for row_id, (t, target_contexts) in enumerate(target2contexts.items()):

        total_f += len(target_contexts)

        # make target representation
        c2f = Counter(target_contexts)
        for c, f in c2f.items():
            try:
                col_id = context2col_id[c]
            except KeyError:  # context is not in contexts_all (includes intersection of ctl and exp contexts)
                pass
            else:
                res[row_id, col_id] = f
                collected_f += f

        if collected_f > max_sum:
            break

    print(f'Collected {collected_f:,}/{total_f:,} co-occurrences')

    # check each representation has information
    num_zero_rows = np.sum(~res.any(axis=1))
    if num_zero_rows > 0:
        print(f'WARNING: Found {num_zero_rows} targets with empty rows (targets that occur with non-shared contexts')
    # this assertion fails when only those context types are included that are shared by exp and ctl targets.
    # some targets only occur with those infrequent, non-shared contexts, and so do not have any counts in the matrix

    return res


def feature_selection(params,
                      targets_exp: SortedSet,
                      targets_ctl: SortedSet,
                      doc: Doc,
                      max_sum: int,
                      ):
    """
    collect contexts, make features, and compute mutual information between contexts and category (exp vs. ctl)

    features are the proportion of times a target co-occurs with a given POS sequence
    """

    # get experimental target contexts
    target_exp2contexts, context_exp2f = get_target_contexts(targets_exp,
                                                             doc,
                                                             direction=params.direction,
                                                             context_size=CONTEXT_SIZE,
                                                             preserve_order=True,
                                                             )
    # get control target contexts
    target_ctl2contexts, context_ctl2f = get_target_contexts(targets_ctl,
                                                             doc,
                                                             direction=params.direction,
                                                             context_size=CONTEXT_SIZE,
                                                             preserve_order=True,
                                                             )
    # make sure that the same contexts are in both feature matrices (an in the same order)
    contexts_shared = SortedSet(context_exp2f.copy().keys())
    contexts_shared.intersection_update(context_ctl2f.keys())

    # make features
    target_exp_rep_mat = make_target_vectors(target_exp2contexts, contexts_shared, max_sum).sum(0, keepdims=True)
    target_ctl_rep_mat = make_target_vectors(target_ctl2contexts, contexts_shared, max_sum).sum(0, keepdims=True)
    print('shape of exp target features={}'.format(target_exp_rep_mat.shape))
    print('shape of ctl target features={}'.format(target_ctl_rep_mat.shape))

    # feature selection: which contexts are most diagnostic of category membership?
    X = np.vstack((target_exp_rep_mat, target_ctl_rep_mat))
    y = np.hstack((
        np.ones(len(target_exp_rep_mat), dtype=np.int) * 1,
        np.ones(len(target_ctl_rep_mat), dtype=np.int) * 0
    ))

    res, p_values = chi2(X, y)  # also returns p values

    # note: large chi2 values do not differentiate between features that diagnose exp vs. ctl or vice versa.
    # lfeatures with large chi2 are simply useful for making the distinction

    # print most diagnostic contexts
    print('----------------|most diagnostic|------------------------------------------------')
    for i in np.argsort(res)[::-1][:20]:
        c = contexts_shared[i]
        print(f'context={" ".join(c):<32} '
              f'chi2={int(res[i]):>12,} '
              f'p-val={p_values[i]:.6f} '
              f'f-exp={context_exp2f[c]:>6,} '
              f'f-ctl={context_ctl2f[c]:>6,} '
              f'{"diagnostic of EXP" if context_exp2f[c] > context_ctl2f[c] else ""}')
    # print('----------------|least diagnostic|-----------------------------------------------')
    # for i in np.argsort(res)[::-1][-10:]:
    #     c = contexts_shared[i]
    #     print(f'context={" ".join(c):<32} '
    #           f'chi2={int(res[i]):>12,} '
    #           f'p-val={p_values[i]:.6f} '
    #           f'f-exp={context_exp2f[c]:>6,} '
    #           f'f-ctl={context_ctl2f[c]:>6,} ')


for params in Conditions.all():

    age2doc = prepare_data(params)  # returns spacy Doc objects
    targets_exp, targets_ctl = make_targets(params, age2doc)

    params = attr.evolve(params,
                         age='combined',
                         direction='l'
                         )
    print(params)

    # for all ages
    doc_combined = Doc.from_docs([d for d in age2doc.values()])
    feature_selection(params, targets_exp, targets_ctl, doc_combined, max_sum=500_000)

    # for each age
    for age, doc in sorted(age2doc.items(), key=lambda i: i[0]):

        params = attr.evolve(params,
                             age=age,
                             direction='l'
                             )
        print(params)

        feature_selection(params, targets_exp, targets_ctl, doc, max_sum=81_000)
