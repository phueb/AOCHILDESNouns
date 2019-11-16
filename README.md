# Word-Play

Research code for analyzing text corpora with a particular emphasis on complexity and categories.

## Custom Dependencies

* [Preppy](https://github.com/phueb/Preppy): prepares text files for analysis
* [CategoryEval](https://github.com/phueb/CategoryEval): has probe words for testing

## Terminology

* Probes: a set of words with known category membership used for testing
* Window: a multi-word sequence form corpus; consists of a left-context and target word
* Context: a multi-word sequence (typically up to 6) preceding a target word
* Target: a single word that follows a context
* Target-by-Context Matrix: a matrix containing counts of co-occurrences between a context (rows) and target (columns)