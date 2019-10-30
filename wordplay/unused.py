

def probe_num_periods_in_context_list(self):
    result = []
    for probe in self.probe_store.types:
        num_periods = self.probe_context_terms_dict[probe].count('.')
        result.append(num_periods / len(self.probe_context_terms_dict[probe]))
    return result


def probe_tag_entropy_list(self):
    result = []
    for probe in self.probe_store.types:
        tag_entropy = scipy.stats.entropy(list(self.train_terms.term_tags_dict[probe].values()))
        result.append(tag_entropy)
    return result


def probe_context_overlap_list(self):
    # probe_context_set_d
    probe_context_set_d = {probe: set() for probe in self.probe_store.types}
    for probe in self.probe_store.types:
        context_set = set(self.probe_context_terms_dict[probe])
        probe_context_set_d[probe] = context_set
    # probe_overlap_list
    result = []
    for probe in self.probe_store.types:
        set_a = probe_context_set_d[probe]
        num_overlap = 0
        for set_b in probe_context_set_d.values():
            num_overlap += len(set_a & set_b)
        probe_overlap = num_overlap / len(set_a)
        result.append(probe_overlap)
    return result


def probe_context_terms_dict(self):
    print('Making probe_context_terms_dict...')
    result = {probe: [] for probe in self.probe_store.types}
    for n, t in enumerate(self.train_terms.tokens):
        if t in self.probe_store.types:
            context = [term for term in self.train_terms.tokens[n - self.params.window_size:n]]
            result[t] += context
    return result


def get_terms_related_to_cat(self, cat):
    cat_probes = self.probe_store.cat_probe_list_dict[cat]
    term_hit_dict = {term: 0 for term in self.train_terms.types}
    for n, term in enumerate(self.train_terms.tokens):
        loc = max(0, n - self.params.window_size)
        if any([term in cat_probes for term in self.train_terms.tokens[loc: n]]):
            term_hit_dict[term] += 1
    result = list(zip(*sorted(term_hit_dict.items(),
                              key=lambda i: i[1] / self.train_terms.w2f[i[0]])))[0]
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
    result = sorted(set(ts), key=lambda t: c[t] / self.train_terms.w2f[t])
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
