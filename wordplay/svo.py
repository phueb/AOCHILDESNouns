from spacy.tokens import Span
from spacy.parts_of_speech import CONJ, DET, NOUN, VERB
import itertools

SUBJ_DEPS = {"agent", "csubj", "csubjpass", "expl", "nsubj", "nsubjpass"}
OBJ_DEPS = {"attr", "dobj", "dative", "oprd"}
AUX_DEPS = {"aux", "auxpass", "neg"}


def get_main_verbs_of_sent(sent):
    """Return the main (non-auxiliary) verbs in a sentence."""
    return [
        tok for tok in sent if tok.pos == VERB and tok.dep_ not in AUX_DEPS
    ]


def get_subjects_of_verb(verb):
    """Return all subjects of a verb according to the dependency parse."""
    subjs = [tok for tok in verb.lefts if tok.dep_ in SUBJ_DEPS]
    # get additional conjunct subjects
    subjs.extend(tok for subj in subjs for tok in _get_conjuncts(subj))
    return subjs


def get_objects_of_verb(verb):
    """
    Return all objects of a verb according to the dependency parse,
    including open clausal complements.
    """
    objs = [tok for tok in verb.rights if tok.dep_ in OBJ_DEPS]
    # get open clausal complements (xcomp)
    objs.extend(tok for tok in verb.rights if tok.dep_ == "xcomp")
    # get additional conjunct objects
    objs.extend(tok for obj in objs for tok in _get_conjuncts(obj))
    return objs


def _get_conjuncts(tok):
    """
    Return conjunct dependents of the leftmost conjunct in a coordinated phrase,
    e.g. "Burton, [Dan], and [Josh] ...".
    """
    return [right for right in tok.rights if right.dep_ == "conj"]


def get_span_for_compound_noun(noun):
    """
    Return document indexes spanning all (adjacent) tokens
    in a compound noun.
    """
    min_i = noun.i - sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ == "compound", reversed(list(noun.lefts))
        )
    )
    return (min_i, noun.i)


def get_span_for_verb_auxiliaries(verb):
    """
    Return document indexes spanning all (adjacent) tokens
    around a verb that are auxiliary verbs or negations.
    """
    min_i = verb.i - sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ in AUX_DEPS, reversed(list(verb.lefts))
        )
    )
    max_i = verb.i + sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ in AUX_DEPS, verb.rights
        )
    )
    return (min_i, max_i)


def subject_verb_object_triples(doc):
    """
    Extract an ordered sequence of subject-verb-object (SVO) triples from a
    spacy-parsed doc. Note that this only works for SVO languages.

    Args:
        doc (:class:`spacy.tokens.Doc` or :class:`spacy.tokens.Span`)

    Yields:
        Tuple[:class:`spacy.tokens.Span`]: The next 3-tuple of spans from ``doc``
        representing a (subject, verb, object) triple, in order of appearance.
    """
    # TODO: What to do about questions, where it may be VSO instead of SVO?
    # TODO: What about non-adjacent verb negations?
    # TODO: What about object (noun) negations?
    if isinstance(doc, Span):
        sents = [doc]
    else:  # spacy.Doc
        sents = doc.sents

    for sent in sents:
        start_i = sent[0].i

        verbs = get_main_verbs_of_sent(sent)
        for verb in verbs:
            subjs = get_subjects_of_verb(verb)
            if not subjs:
                continue
            objs = get_objects_of_verb(verb)
            if not objs:
                continue

            # add adjacent auxiliaries to verbs, for context
            # and add compounds to compound nouns
            verb_span = get_span_for_verb_auxiliaries(verb)
            verb = sent[verb_span[0] - start_i : verb_span[1] - start_i + 1]
            for subj in subjs:
                subj = sent[
                    get_span_for_compound_noun(subj)[0]
                    - start_i : subj.i
                    - start_i
                    + 1
                ]
                for obj in objs:
                    if obj.pos == NOUN:
                        span = get_span_for_compound_noun(obj)
                    elif obj.pos == VERB:
                        span = get_span_for_verb_auxiliaries(obj)
                    else:
                        span = (obj.i, obj.i)
                    obj = sent[span[0] - start_i : span[1] - start_i + 1]

                    yield (subj.text, verb.text, obj.text)