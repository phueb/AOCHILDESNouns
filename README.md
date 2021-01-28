# AbstractFirst

Research code for replicating experiments in a future paper.

## Research Question

Is the noun co-occurrence structure in speech to younger children less lexically specific (abstract) ? 

Nouns were obtained from by:
- collecting all words tagged by `spacy` as noun in a `spacy`-tokenized American-English CHILDES corpus
- excluding words which are not among 4k most frequent words in corpus
- excluding onomatopeia, interjections, single characters, gerunds, proper names
- misspelled words

## Compatibility

Developed on Ubuntu 16.04 and Python 3.7