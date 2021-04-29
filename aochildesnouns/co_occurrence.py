from typing import List, Tuple, Dict
from spacy.tokens import Doc
from scipy import sparse
from sortedcontainers import SortedSet


from aochildesnouns.params import Params
from aochildesnouns import configs


class CoData:
    """store co-occurrence data, for left and right directions separately"""

    def __init__(self,
                 row_ids_l: List[int],
                 col_ids_l: List[int],
                 row_ids_r: List[int],
                 col_ids_r: List[int],
                 lw2id: Dict[str, int],
                 cw2id: Dict[str, int],
                 rw2id: Dict[str, int],
                 ):

        assert len(row_ids_l) == len(col_ids_l)
        assert len(row_ids_r) == len(col_ids_r)

        self.row_ids_l = row_ids_l
        self.col_ids_l = col_ids_l

        self.row_ids_r = row_ids_r
        self.col_ids_r = col_ids_r

        self.row_ids_b = row_ids_l + row_ids_r
        self.col_ids_b = col_ids_l + [i + len(col_ids_l) for i in col_ids_r]

        self.lw2id = lw2id
        self.cw2id = cw2id
        self.rw2id = rw2id

    def as_matrix(self,
                  direction: str,
                  ) -> sparse.coo_matrix:

        if direction == 'l':
            data = [1] * len(self.col_ids_l)
            row_ids = self.row_ids_l
            col_ids = self.col_ids_l
            assert max(self.col_ids_l) + 1 == len(set(self.col_ids_l))  # check that col ids are consecutive
        elif direction == 'r':
            data = [1] * len(self.col_ids_r)
            row_ids = self.row_ids_r
            col_ids = self.col_ids_r
            assert max(self.col_ids_r) + 1 == len(set(self.col_ids_r))  # check that col ids are consecutive
        elif direction == 'b':
            data = [1] * len(self.col_ids_l + self.col_ids_r)
            row_ids = self.row_ids_l + self.row_ids_r
            col_ids = self.col_ids_l + self.row_ids_r
        else:
            raise AttributeError('Invalid arg')

        res = sparse.coo_matrix((data, (row_ids, col_ids)))

        # check that there are no skipped column ids or zero columns
        res_csc = res.tocsc()
        for col_id in range(res.shape[1]):
            assert col_id in col_ids
            assert res_csc[:, col_id].sum() != 0

        print(f'Co-occurrence matrix has sum={res.sum():,} and shape={res.shape}')

        return res

    def get_words_ordered_by_id(self,
                                direction: str,
                                ) -> Tuple[List[str], List[str]]:
        """get row words, column words (in order in which they correctly label the co-occurrence matrix)"""

        if direction == 'l':
            row_words = [w for w in self.cw2id]
            col_words = [w for w in self.lw2id]
            return row_words, col_words
        elif direction == 'r':
            row_words = [w for w in self.cw2id]
            col_words = [w for w in self.rw2id]
            return row_words, col_words
        elif direction == 'b':
            row_words = [w for w in self.cw2id]
            col_words = [w for w in self.lw2id] + [w for w in self.rw2id]
            return row_words, col_words
        else:
            raise AttributeError('Invalid arg')

    def get_x_y(self,
                direction: str,
                ) -> Tuple[List[int], List[int]]:
        """get realizations of two random variables, x, and y, corresponding to row and column of co matrix"""
        if direction == 'l':
            return self.row_ids_l, self.col_ids_l
        elif direction == 'r':
            return self.row_ids_r, self.col_ids_r
        elif direction == 'b':
            return self.row_ids_b, self.col_ids_b
        else:
            raise AttributeError('Invalid arg')

    def get_x_y_z(self):
        """get realizations of three random variables, x, y, z, where y are left and z are right context type ids"""
        assert self.row_ids_r == self.row_ids_l
        return self.row_ids_r, self.col_ids_l, self.col_ids_r


def collect_left_and_right_co_occurrences(doc: Doc,
                                          targets: SortedSet,
                                          params: Params,
                                          ) -> CoData:
    """
    collect co-occurrences in format suitable for scipy.sparse.coo.

    note: collects left, right and "both" co-occurrences.
    in the latter case, both a right and left co-occurrence are recorded separately.
    """

    max_sum = configs.Data.max_sum[params.targets_control]
    assert isinstance(max_sum, int) or max_sum is None

    cw2id = {}
    lw2id = {}
    rw2id = {}
    row_ids_l = []
    row_ids_r = []
    col_ids_l = []
    col_ids_r = []
    num_targets = 0  # just for reference - print this
    for n in range(len(doc) - 2):

        if n == 0:
            continue
        if doc[n].text not in targets:
            continue
        if not params.targets_control and doc[n].tag_ not in params.tags:  # do not filter when using control targets
            continue

        # handle punctuation
        lwo = 1  # left word offset
        rwo = 1  # right word offset
        while True:
            is_break = True
            # left word, center word, right word
            if params.lemmas:
                lw = doc[n - lwo].lemma_
                cw = doc[n].lemma_
                rw = doc[n + rwo].lemma_
            else:
                lw = doc[n - lwo].text
                cw = doc[n].text
                rw = doc[n + rwo].text

            if lw in configs.Data.punctuation:
                if params.punctuation == 'remove':
                    lwo += 1
                    is_break = False
                elif params.punctuation == 'merge':
                    lw = configs.Data.eos
            if rw in configs.Data.punctuation:
                if params.punctuation == 'remove':
                    rwo += 1
                    is_break = False
                elif params.punctuation == 'merge':
                    rw = configs.Data.eos

            if is_break:
                break

        if max_sum is not None and len(row_ids_l) < max_sum:
            # collect left co-occurrence
            row_ids_l.append(cw2id.setdefault(cw, len(cw2id)))
            col_ids_l.append(lw2id.setdefault(lw, len(lw2id)))

            # collect right co-occurrence
            row_ids_r.append(cw2id.setdefault(cw, len(cw2id)))
            col_ids_r.append(rw2id.setdefault(rw, len(rw2id)))

        # for reference - deciding max_sum
        num_targets += 1

    print(f'Num total targets={num_targets:,}. Collected {len(row_ids_l):,} co-occurrences')

    return CoData(row_ids_l,
                  col_ids_l,
                  row_ids_r,
                  col_ids_r,
                  lw2id,
                  cw2id,
                  rw2id,
                  )
