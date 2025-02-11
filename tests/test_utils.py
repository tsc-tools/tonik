import numpy as np

from tonik.utils import extract_consecutive_integers


def test_extract_consecutive_integers():
    nums = [1, 2, 3, 5, 6, 7, 8, 10]
    assert extract_consecutive_integers(
        nums) == [[1, 2, 3], [5, 6, 7, 8], [10]]
    assert extract_consecutive_integers([1]) == [[1]]
    assert extract_consecutive_integers(np.array([1, 2, 4])) == [[1, 2], [4]]
