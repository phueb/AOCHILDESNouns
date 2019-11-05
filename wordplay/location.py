from itertools import chain


def make_w2location(tokens):
    print('Making w2location...')
    result = {item: [] for item in set(tokens)}
    for loc, term in enumerate(tokens):
        result[term].append(loc)
    return result


def term_unordered_locs_dict(self):  # keep this for fast calculation where order doesn't matter
    print('Making term_unordered_locs_dict...')
    result = {item: [] for item in self.train_terms.types}
    for loc, term in enumerate(self.train_terms.tokens):
        result[term].append(loc)
    return result


def term_avg_reordered_loc_dict(self):
    result = {}
    for term, locs in self.w2location.items():
        result[term] = np.mean(locs)
    return result


def term_avg_unordered_loc_dict(self):
    result = {}
    for term, locs in self.term_unordered_locs_dict.items():
        result[term] = np.mean(locs)
    return result


def calc_avg_reordered_loc(self, term):
    result = int(self.term_avg_reordered_loc_dict[term])
    return result


def calc_avg_unordered_loc(self, term):
    result = int(self.term_avg_unordered_loc_dict[term])
    return result


def calc_lateness(self, term, is_probe, reordered=True):
    fn = self.calc_avg_reordered_loc if reordered else self.calc_avg_unordered_loc
    if is_probe:
        ref_loc = self.probes_reordered_loc * 2 if reordered else self.probes_unordered_loc * 2
    else:
        ref_loc = self.train_terms.num_tokens
    result = round(fn(term) / ref_loc, 2)
    return result


def probe_lateness_dict(self):
    print('Making probe_lateness_dict...')
    probe_latenesses = []
    for probe in self.probe_store.types:
        probe_lateness = self.calc_lateness(probe, is_probe=True)
        probe_latenesses.append(probe_lateness)
    result = {probe: round(np.asscalar(np.mean(probe_lateness)), 2)
              for probe_lateness, probe in zip(probe_latenesses, self.probe_store.types)}
    return result


def probes_reordered_loc(self):
    n_sum = 0
    num_ns = 0
    for n, term in enumerate(self.reordered_tokens):
        if term in self.probe_store.types:
            n_sum += n
            num_ns += 1
    result = n_sum / num_ns
    return result


def probes_unordered_loc(self):
    loc_sum = 0
    num_locs = 0
    for loc, term in enumerate(self.train_terms.tokens):
        if term in self.probe_store.types:
            loc_sum += loc
            num_locs += 1
    result = loc_sum / num_locs
    return result


def midpoint_loc(self):
    result = self.train_terms.num_tokens // 2
    return result


def split_probes_by_loc(self, num_splits, is_reordered=False):
    if is_reordered:
        d = self.term_avg_reordered_loc_dict
    else:
        d = self.term_avg_unordered_loc_dict
    probe_loc_pairs = [(probe, loc) for probe, loc in d.items()
                       if probe in self.probe_store.types]
    sorted_probe_loc_pairs = sorted(probe_loc_pairs, key=lambda i: i[1])
    num_in_split = self.probe_store.num_probes // num_splits
    for split_id in range(num_splits):
        start = num_in_split * split_id
        probes, _ = zip(*sorted_probe_loc_pairs[start: start + num_in_split])
        yield probes


def make_locs_xy(self, terms, num_bins=20):
    item_locs_l = [self.term_unordered_locs_dict[i] for i in terms]  # TODO use reordered locs?
    locs_l = list(chain(*item_locs_l))
    hist, b = np.histogram(locs_l, bins=num_bins)
    result = (b[:-1], np.squeeze(hist))
    return result


def calc_loc_asymmetry(self, term, num_bins=200):  # num_bins=200 for terms
    (x, y) = self.make_locs_xy([term], num_bins=num_bins)
    y_fitted = self.fit_line(x, y)
    result = linregress(x / self.train_terms.num_tokens, y_fitted)[0]  # divide x to increase slope
    return result