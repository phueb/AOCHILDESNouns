# AbstractFirst

Research code for replicating experiments in a future paper.

## Research Question

Is the noun co-occurrence structure in speech to younger children less lexically specific (abstract) ? 


## Technical Notes

### Nouns

Nouns were obtained from by:
- collecting all words tagged by `spacy` as noun in a `spacy`-tokenized American-English CHILDES corpus
- excluding words which are not among 4k most frequent words in corpus
- excluding onomatopeia, interjections, single characters, gerunds, proper names
- misspelled words

### What does the largest singular mean?

The first singular dimension of a lexical co-occurrence matrix can be thought of as representing a 
distribution of lexical frequencies, which best fits the observed frequencies of context types 
(each context type is associated with a unique column in the matrix) of __all__ target types 
(each associated with a unique row). 
Its associated singular value indicates how well this single distribution describes the full co-occurrence matrix.
This value will be larger when targets have similar context type distributions, 
and will be smaller if targets have dissimilar context type distributions.
Thus, a larger the fist singular value, relative to the other singular values, 
means that the rows in the co-occurrence matrix, when projected on the first singular dimension,
will be better approximations of the original co-occurrence matrix.

## Compatibility

Developed on Ubuntu 16.04 and Python 3.7