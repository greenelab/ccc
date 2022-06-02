import pytest
import numpy as np

from ccc.pytorch.core import unravel_index_2d


def test_unravel_index_2d_square_simple():
    shape = (2, 2)
    assert unravel_index_2d(0, shape) == (0, 0)
    assert unravel_index_2d(1, shape) == (0, 1)
    assert unravel_index_2d(2, shape) == (1, 0)
    assert unravel_index_2d(3, shape) == (1, 1)


def test_unravel_index_2d_rect_simple():
    shape = (2, 3)
    assert unravel_index_2d(0, shape) == (0, 0)
    assert unravel_index_2d(1, shape) == (0, 1)
    assert unravel_index_2d(2, shape) == (0, 2)
    assert unravel_index_2d(3, shape) == (1, 0)
    assert unravel_index_2d(4, shape) == (1, 1)
    assert unravel_index_2d(5, shape) == (1, 2)

    shape = (1, 4)
    assert unravel_index_2d(0, shape) == (0, 0)
    assert unravel_index_2d(1, shape) == (0, 1)
    assert unravel_index_2d(2, shape) == (0, 2)
    assert unravel_index_2d(3, shape) == (0, 3)

    shape = (4, 1)
    assert unravel_index_2d(0, shape) == (0, 0)
    assert unravel_index_2d(1, shape) == (1, 0)
    assert unravel_index_2d(2, shape) == (2, 0)
    assert unravel_index_2d(3, shape) == (3, 0)


def test_unravel_index_2d_square0():
    x = np.array([[0, 7], [-5, 6.999]])
    x_max_idx = np.argmax(x, axis=None)
    assert x_max_idx == 1

    expected_idx = np.unravel_index(x_max_idx, x.shape)
    observed_idx = unravel_index_2d(x_max_idx, x.shape)

    assert expected_idx == observed_idx == (0, 1)


def test_unravel_index_2d_square1():
    x = np.array([[0, 7], [-5, 7.01]])
    x_max_idx = np.argmax(x, axis=None)
    assert x_max_idx == 3

    expected_idx = np.unravel_index(x_max_idx, x.shape)
    observed_idx = unravel_index_2d(x_max_idx, x.shape)

    assert expected_idx == observed_idx == (1, 1)


def test_unravel_index_2d_square_all_equal():
    x = np.array([[7.0, 7.0], [7.0, 7.0]])
    x_max_idx = np.argmax(x, axis=None)
    assert x_max_idx == 0

    expected_idx = np.unravel_index(x_max_idx, x.shape)
    observed_idx = unravel_index_2d(x_max_idx, x.shape)

    assert expected_idx == observed_idx == (0, 0)


def test_unravel_index_2d_rect():
    x = np.array([[0, 7, -5.6], [8.1, 6.999, 0]])
    x_max_idx = np.argmax(x, axis=None)
    assert x_max_idx == 3

    expected_idx = np.unravel_index(x_max_idx, x.shape)
    observed_idx = unravel_index_2d(x_max_idx, x.shape)

    assert expected_idx == observed_idx == (1, 0)


def test_unravel_index_index_out_of_bounds():
    with pytest.raises(ValueError):
        unravel_index_2d(6, (2, 3))


def test_unravel_index_non_2d():
    with pytest.raises(ValueError):
        unravel_index_2d(0, (2, 3, 4))
