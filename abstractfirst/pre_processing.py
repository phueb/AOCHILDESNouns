import numpy as np
from typing import List, Dict, Tuple
import spacy
from spacy.tokens import Doc, DocBin
from sortedcontainers import SortedSet
from spacy.tokens.doc import Doc

from abstractfirst import configs
from abstractfirst.params import Params

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


EXCLUDED_AGE = 'EXCLUDED'


def load_transcripts(params: Params):

    corpus_path = configs.Dirs.corpora / f'{params.corpus_name}.txt'
    corpus_text = corpus_path.read_text(encoding='utf-8')

    if params.punctuation == 'keep':
        print('Keeping punctuation as-is')

    elif params.punctuation == 'merge':
        print('Merging punctuation into one symbol')
        for symbol in {'.', '!', '?'}:
            corpus_text = corpus_text.replace(symbol, '[EOS]')

    elif params.punctuation == 'remove':
        print('Removing punctuation')
        for symbol in {'. ', '! ', '? '}:
            corpus_text = corpus_text.replace(symbol, '')

    else:
        raise AttributeError('Invalid arg to punctuation')

    # remove duplicated whitespace
    corpus_text = corpus_text.strip()
    while '  ' in corpus_text:
        corpus_text = corpus_text.replace('  ', ' ')

    transcripts: List[str] = corpus_text.split('\n')[:-1]

    return transcripts


def load_ages(params: Params,
              ) -> List[str]:
    """convert ages into strings e.g. '0-900days' """
    ages_path = configs.Dirs.corpora / f'{params.corpus_name}_ages.txt'
    ages_text = ages_path.read_text(encoding='utf-8')
    ages = [float(ai) for ai in ages_text.split()]

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
                 ) -> Dict[str, Doc]:
    """
    return a single spacy doc for each age.

    warning: currently, only one corpus binary is saved to disk, with punctuation intact.
    when params.punctuation is anything but "keep", the raw corpus will be loaded,
     in order for punctuation to be processed as specified by params.punctuation, and POS-tagged
    """

    # try loading transcripts from disk
    fn = params.corpus_name + '.spacy'
    bin_path = configs.Dirs.corpora / fn
    if bin_path.exists() and params.punctuation == 'keep':
        doc_bin = DocBin().from_disk(bin_path)
        docs = list(doc_bin.get_docs(nlp.vocab))
    # load raw transcripts + process them
    else:
        print(f'WARNING: Did not find binary file associated with {params.corpus_name}. Preprocessing corpus...')
        transcripts = load_transcripts(params)
        docs: List[Doc] = [doc for doc in nlp.pipe(transcripts)]
        # only save to disk if we know that punctuation has not been modified
        if params.punctuation == 'keep':
            doc_bin = DocBin(docs=docs)
            doc_bin.to_disk(bin_path)

    # group docs by age
    ages = load_ages(params)
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
