import spacy
from typing import List, Set
import string

from wordplay import config
from wordplay.word_sets import excluded

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

pos2tags = {
    'NOUN': {'NN', 'NNS'},
    'VERB': {'MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},
    'ADJ': {'AFX', 'JJ', 'JJR', 'JJS'},
    'P-NOUN': {'NNP', 'NNPS'},
    'ADV': {'RB', 'RBR', 'RBS', 'WRB'},
    'PRON': {'PRP', 'PRP$', 'WP', 'WP$', 'EX'},
    'ADP': {'IN'},
    'conj': {'CC'},
    'INTJ': {'UH'},
    'DET': {'DT', 'PDT', 'WDT'},
    'particle': {'POS', 'RP', 'TO'},
    'punct': {',', ':', '.', "''", 'HYPH', 'NFP'},
}

tag2pos = {

    'NN': 'NOUN',
    'NNS': 'NOUN',

    'MD': 'VERB',
    'VB': 'VERB',
    'VBD': 'VERB',
    'VBG': 'VERB',
    'VBN': 'VERB',
    'VBP': 'VERB',
    'VBZ': 'VERB',

    'AFX': 'ADJ',
    'JJ': 'ADJ',
    'JJR': 'ADJ',
    'JJS': 'ADJ',

    'NNP': 'P-NOUN',
    'NNPS': 'P-NOUN',

    'RB': 'ADV',
    'RBR': 'ADV',
    'RBS': 'ADV',
    'WRB': 'ADV',

    'PRP': 'PRON',
    'PRP$': 'PRON',
    'WP': 'PRON',
    'WP$': 'PRON',
    'EX': 'PRON',

    'IN': 'ADP',
    'CC': 'conj',
    'UH': 'INTJ',

    'PDT': 'DET',
    'WDT': 'DET',
    'DT': 'DET',

    'POS': 'particle',
    'RP': 'particle',
    'TO': 'particle',

    '.': 'punct',
    ',': 'punct',
    ':': 'punct',
    "'": 'punct',
    'HYPH': 'punct',
    'NFP': 'punct',

    'CD': 'number',

}


excluded_set = set(string.printable.split() + config.Symbols.all)


def make_pos_words(vocab: List[str],
                   pos: str,
                   ) -> Set[str]:
    print(f'Making list of {pos} words in vocabulary...')
    res = set()
    for w in vocab:
        pos_ = nlp(w)[0].pos_
        if pos_ == pos and w not in excluded_set:
            res.add(w)

    assert len(res) != 0
    print(f'Found {len(res)} {pos}s')

    return res


def load_pos_words(file_name: str
                   ) -> Set[str]:
    p = config.Dirs.words / f'{file_name}.txt'
    res = p.read_text().split('\n')
    return set(res)