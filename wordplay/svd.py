from scipy import sparse


def make_term_by_window_co_occurrence_mat(self, tokens=None, start=None, end=None, window_size=7,
                                          only_probes_in_x=False):
    """
    terms are in cols, windows are in rows.
    y_words windows, x_words are terms.
    y_words always occur after y_words in the input.
    this format matches that used in TreeTransitions
    """

    # tokens
    if tokens is None:
        if start is not None and end is not None:
            tokens = self.train_terms.tokens[start:end]
        else:
            raise ValueError('Need either "tokens" or "start" and "end".')

    # x_words
    x_words = self.probe_store.types if only_probes_in_x else sorted(set(tokens))
    num_xws = len(x_words)
    xw2col_id = {t: n for n, t in enumerate(x_words)}
    # windows
    windows_in_order = self.get_sliding_windows(window_size, tokens)
    unique_windows = sorted(set(windows_in_order))
    num_unique_windows = len(unique_windows)
    window2row_id = {w: n for n, w in enumerate(unique_windows)}

    # make sparse matrix (y_words in rows, windows in cols)
    shape = (num_unique_windows, num_xws)
    print('Making term-by-window co-occurrence matrix2 with shape={}...'.format(shape))
    data = []
    row_ids = []
    cold_ids = []
    mat_loc2freq = {}  # needed to keep track of freq
    for n, window in enumerate(windows_in_order[:-window_size]):
        # row_id + col_id
        row_id = window2row_id[window]
        next_window = windows_in_order[n + 1]
        next_term = next_window[-1]  # -1 is correct because windows slide by 1 word
        try:
            col_id = xw2col_id[next_term]
        except KeyError:  # only_probes_in_x
            continue
        # freq
        try:
            freq = mat_loc2freq[(row_id, col_id)]
        except KeyError:
            mat_loc2freq[(row_id, col_id)] = 1
            freq = 1
        else:
            mat_loc2freq[(row_id, col_id)] += 1
        # collect
        row_ids.append(row_id)
        cold_ids.append(col_id)
        data.append(freq)

    # make sparse matrix once (updating it is expensive)
    res = sparse.csr_matrix((data, (row_ids, cold_ids)), shape=shape)

    print('term-by-window co-occurrence matrix2 has sum={:,}'.format(res.sum()))

    y_words = unique_windows
    return res, x_words, y_words
