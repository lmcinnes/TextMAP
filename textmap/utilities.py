from collections.abc import Iterable
import itertools
import numpy as np


def flatten(list_of_seq):
    assert isinstance(list_of_seq, Iterable)
    if len(list_of_seq) == 0:
        return list_of_seq
    if type(list_of_seq[0]) in (list, tuple, np.ndarray):
        return tuple(itertools.chain.from_iterable(list_of_seq))
    else:
        return list_of_seq


def flatten_list(list_of_seq):
    assert isinstance(list_of_seq, Iterable)
    if len(list_of_seq) == 0:
        return list_of_seq
    if type(list_of_seq[0]) in (list, tuple, np.ndarray):
        return list(itertools.chain.from_iterable(list_of_seq))
    else:
        return list_of_seq
