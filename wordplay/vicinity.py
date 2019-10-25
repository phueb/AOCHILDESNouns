


def get_terms_related_to_cat(self, cat):
    cat_probes = self.probe_store.cat_probe_list_dict[cat]
    term_hit_dict = {term: 0 for term in self.train_terms.types}
    for n, term in enumerate(self.train_terms.tokens):
        loc = max(0, n - self.params.window_size)
        if any([term in cat_probes for term in self.train_terms.tokens[loc: n]]):
            term_hit_dict[term] += 1
    result = list(zip(*sorted(term_hit_dict.items(),
                              key=lambda i: i[1] / self.train_terms.term_freq_dict[i[0]])))[0]
    return result


def get_term_set_prop_near_terms(self, terms, dist=1):
    ts = []
    for loc, t in enumerate(self.train_terms.tokens):
        if t in terms:
            try:
                ts.append(self.train_terms.tokens[loc + dist])
            except IndexError:  # hit end or start of tokens
                print('rnnlab: Invalid tokens location: {}'.format(loc + dist))
                continue
    c = Counter(ts)
    result = sorted(set(ts), key=lambda t: c[t] / self.train_terms.term_freq_dict[t])
    return result


def get_terms_near_term(self, term, dist=1):
    result = []
    for loc in self.term_reordered_locs_dict[term]:
        try:
            result.append(self.reordered_tokens[loc + dist])
        except IndexError:  # location too early or too late
            pass
    return result


def cat_common_successors_dict(self, num_most_common=60):  # TODO vary num_most_common
    cat_successors_dict = {cat: [] for cat in self.probe_store.cats}
    for cat, cat_probes in self.probe_store.cat_probe_list_dict.items():
        for cat_probe in cat_probes:
            terms_near_cat_probe = self.get_terms_near_term(cat_probe)
            cat_successors_dict[cat] += terms_near_cat_probe
    # filter
    result = {cat: list(zip(*Counter(cat_successors).most_common(num_most_common)))[0]
              for cat, cat_successors in cat_successors_dict.items()}
    return result


def probes_common_successors_dict(self, num_most_common=60):  # TODO vary num_most_common
    probes_successors = []
    for probe in self.probe_store.types:
        terms_near_probe = self.get_terms_near_term(probe)
        probes_successors += terms_near_probe
    # filter
    result = list(zip(*Counter(probes_successors).most_common(num_most_common)))[0]
    return result
