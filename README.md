# StartingEntropic

Research code for replicating experiments in an upcoming paper.

## Research Question

Do nouns form a better category in speech to younger vs. older children? 

One way to study the structure of data is to decompose it into linearly separable and orthogonal dimensions, 
which can be done with SVD.
Below is a visualisation of the noun-co-occurrence matrix of speech to children under 900 days old, 
projected on the first, then first + second, then first + second + third, ... singular dimensions, 
 incrementing with each new animation frame.

<div align="center">
 <img src="animations/readme1.gif" width="600">
</div>


## Replication

Clone the repository, then install requirements (preferably into a virtual Python3.7 environment):
```bash
pip install -r requirements.txt
```

Optionally edit the conditions to replicate in `startingentropic/params.py`, and then:

```bash
python3 main.py
```

## Technical Notes

### Nouns

Nouns were obtained from by:
- collecting all words tagged by `spacy` as noun in a `spacy`-tokenized American-English CHILDES corpus
- excluding words which are not among 4k most frequent words in corpus
- excluding onomatopeia, interjections, single characters, gerunds, proper names
- misspelled words

### What does the largest singular value mean?

Assume we are talking about a lexical co-occurrence matrix, 
where each context type is associated with a unique column in the matrix,
and target words are associated with unique rows. 

The first singular dimension of a lexical co-occurrence matrix can be thought of as a vector 
whose elements are lexical frequencies, which best fits the observed frequencies of context types across __all__ target types.
Its associated singular value indicates how well this single distribution describes the full co-occurrence matrix.
This value will be larger when _all__ (not just pairwise) targets have similar context type distributions, 
and will be smaller if all targets have dissimilar context type distributions.
Thus, a larger the fist singular value, relative to the other singular values, 
means that the co-occurrence matrix, projected on the first singular dimension,
will be a better approximation of the original co-occurrence matrix.

## Compatibility

Developed on Ubuntu 18.04 and Python 3.7
