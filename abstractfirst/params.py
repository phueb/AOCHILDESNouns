import attr


@attr.s
class Params:
    corpus_name = attr.ib(default='childes-20201026')
    age_step = attr.ib(default=900)
    num_tokens_per_bin = attr.ib(default=2_527_000)     # 2_527_000 is good with params.age_ste=900
    max_sum = attr.ib(default=300_000)    # or None
    targets_name = attr.ib(default='sem-all')
    tags = attr.ib(default={'NN', 'NNS'})
    left_only = attr.ib(default=False)
    right_only = attr.ib(default=True)

    def __str__(self):
        res = ''
        for a in self.__attrs_attrs__:
            if a.name in ['age_step', 'num_tokens_per_bin', 'max_sum']:
                continue
            res += f'{a.name}={a.default}\n'
        return res
