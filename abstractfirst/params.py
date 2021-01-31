import attr
from typing import List, Optional
import itertools

#
# class Conditions:
#     params = attr.ib()
#
#     conditions = {
#         'direction':['l', 'r', 'b']
#
#     def __iter__:
#         for args in itertools.product()


@attr.s
class Params:
    # conditions
    age = attr.ib(default=0)
    tags = attr.ib(default={'NN', 'NNS'})
    direction = attr.ib(default='b')

    corpus_name = attr.ib(default='childes-20201026')
    age_step = attr.ib(default=900)
    num_tokens_per_bin = attr.ib(default=1_000_000)     # 2_527_000 is good with params.age_ste=900
    max_sum = attr.ib(default=300_000)    # or None
    targets_name = attr.ib(default='sem-all')
