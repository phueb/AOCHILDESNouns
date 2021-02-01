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
    age = attr.ib(default='')
    direction = attr.ib(default='')
    merge_punctuation = attr.ib(default=False)
    targets_control = attr.ib(default=False)
    lemmas = attr.ib(default=True)  # TODO implement

    corpus_name = attr.ib(default='childes-20201026')
    num_days = attr.ib(default=1000)  # age range in each age bin - there are always two bins
    max_sum_one_direction = attr.ib(default=88_000)
    targets_name = attr.ib(default='sem-all')
    tags = attr.ib(default={'NN', 'NNS'})
