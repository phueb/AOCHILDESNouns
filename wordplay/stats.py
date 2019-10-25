

"""

# TODO use functions from https://github.com/kristopherkyle/lexical_diversity

"""
from collections import Counter


def calc_entropy(labels):
    num_labels = len(labels)
    probs = np.asarray([count / num_labels for count in Counter(labels).values()])
    result = - probs.dot(np.log2(probs))
    return result


def make_sentence_length_stat(items, is_avg, w_size=10000):
    # make sent_lengths
    last_period = 0
    sent_lengths = []
    for n, item in enumerate(items):
        if item in ['.', '!', '?']:
            sent_length = n - last_period - 1
            sent_lengths.append(sent_length)
            last_period = n
    # rolling window
    df = pd.Series(sent_lengths)
    if is_avg:
        print('Making sentence length rolling average...')
        result = df.rolling(w_size).std().values
    else:
        print('Making sentence length rolling std...')
        result = df.rolling(w_size).mean().values
    return result


def part_entropies(self):
    result = []
    for part in self.reordered_partitions:
        part_entropy = self.calc_entropy(part)
        result.append(part_entropy)
    return result