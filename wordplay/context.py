

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