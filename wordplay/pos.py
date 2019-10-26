import spacy
from typing import List, Set
import string

from wordplay import config

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

pos2tags = {'verb': ['BES', 'HVS', 'MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'noun': ['NN', 'NNS', 'WP'],
            'adverb': ['EX', 'RB', 'RBR', 'RBS', 'WRB'],
            'pronoun': ['PRP'],
            'preposition': ['IN'],
            'conjunction': ['CC'],
            'interjection': ['UH'],
            'determiner': ['DT'],
            'particle': ['POS', 'RP', 'TO'],
            'punctuation': [',', ':', '.', "''", 'HYPH', 'LS', 'NFP'],
            'adjective': ['AFX', 'JJ', 'JJR', 'JJS', 'PDT', 'PRP$', 'WDT', 'WP$'],
            'special': []}

pos2pos_ = {'noun': 'NOUN',
            'verb': 'VERB',
            'preposition': 'ADP'}

excluded_set = set(string.printable.split() + config.Symbols.all)


def make_pos_words(vocab: List[str],
                   pos: str,
                   ) -> Set[str]:
    res = set()
    for w in vocab:
        pos_ = nlp(w)[0].pos_
        if pos_ == pos2pos_[pos] and w not in excluded_set and not w.endswith('s'):
            res.add(w)

    assert len(res) != 0
    print(f'Found {len(res)} {pos}s')

    return res


def load_pos_words(file_name: str
                   ) -> Set[str]:
    p = config.Dirs.words / f'{file_name}.txt'
    res = p.read_text().split('\n')
    return res