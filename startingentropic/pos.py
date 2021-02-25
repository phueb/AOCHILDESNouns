import spacy
import string


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


