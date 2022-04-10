from typing import List, Dict, Tuple
import spacy
from spacy.tokens import Doc, DocBin
from sortedcontainers import SortedSet
from spacy.tokens.doc import Doc

from aochildes.params import AOChildesParams
from aochildes.pipeline import Pipeline
from aochildes.helpers import Transcript

from aochildesnouns import configs
from aochildesnouns.params import Params

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


EXCLUDED_AGE = 'EXCLUDED'


def format_ages(ages: List[float],
                params: Params,
                ) -> List[str]:
    """convert ages into strings e.g. '0-900days' """

    min_age = min(ages)
    max_age = max(ages)

    if min_age + params.num_days > max_age - params.num_days:
        max_num_days = (max_age - min_age) / 2
        raise ValueError(f'num_days cannot be larger than {max_num_days}. would result in overlapping age bins')

    res = []
    age1 = f'{min_age:06.0f}-{min_age + params.num_days:06.0f}days'
    age2 = f'{max_age - params.num_days:06.0f}-{max_age:06.0f}days'

    for ai in ages:
        # get first bin
        if min_age + ai <= params.num_days:
            res.append(age1)
        # get last bin
        elif max_age - ai <= params.num_days:
            res.append(age2)
        else:
            if configs.Data.make_last_bin_larger:  # get more data, but age range of bins are no longer equally large
                res.append(age2)
            else:
                res.append(EXCLUDED_AGE)  # will be excluded

    return res


def prepare_data(params: Params,
                 verbose: bool = True,
                 load_binary: bool = True,
                 ) -> Dict[str, Doc]:
    """
    return a single spacy doc for each age.

    warning: if corpus binary is not on disk already, it will be saved to disk.
    this means the corpus should never be modified - else, the binary will also contain unexpected modifications
    """

    # load transcripts
    aochildes_params = AOChildesParams()
    pipeline = Pipeline(aochildes_params)
    transcripts: List[Transcript] = pipeline.load_age_ordered_transcripts()

    # try loading binary transcripts from disk
    fn = 'aochildes.spacy'  # TODO the same binary file will be loaded even after a change in AOCHILDESParams was made
    bin_path = configs.Dirs.corpora / fn
    if bin_path.exists() and load_binary:
        doc_bin = DocBin().from_disk(bin_path)
        docs = list(doc_bin.get_docs(nlp.vocab))
    # load raw transcripts + process them
    else:
        print(f'WARNING: Did not find binary file. Preprocessing corpus...')
        docs: List[Doc] = [doc for doc in nlp.pipe((t.text for t in transcripts))]
        # WARNING: only save to disk if we know that corpus has not been modified
        doc_bin = DocBin(docs=docs)
        doc_bin.to_disk(bin_path)

    # group docs by age
    ages = format_ages([t.age for t in transcripts], params)
    if len(ages) != len(docs):
        raise RuntimeError(f'Num docs={len(docs)} and num ages={len(ages)}')
    age2docs = {}
    for age in SortedSet(ages):
        if age == EXCLUDED_AGE:
            continue
        docs_at_age = [docs[n] for n, ai in enumerate(ages) if ai == age]
        age2docs[age] = docs_at_age
        if verbose:
            print(f'Processed {len(age2docs[age]):>6} transcripts for age={age}')

    # combine all documents at same age
    age2doc = {}
    for age, docs in age2docs.items():

        doc_combined = Doc.from_docs(docs)
        age2doc[age] = doc_combined
        print(f'Num tokens at age={age} is {len(doc_combined):,}')

    return age2doc
