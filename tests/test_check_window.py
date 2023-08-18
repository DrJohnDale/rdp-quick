import numpy as np
import numpy.testing as np_test
from rdp_quick.check_window import check_windows, check_window


def test_check_window_all_good():
    epsilon = 1.0
    p_arr = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ])
    all_good, max_index = check_window(p_arr, epsilon)
    assert all_good
    assert max_index == 0


def test_check_window_one_bad():
    epsilon = 1.0
    p_arr = np.array([
        [0.0, 0.0],
        [1.0, 10.0],
        [2.0, 0.0],
    ])
    all_good, max_index = check_window(p_arr, epsilon)
    assert not all_good
    assert max_index == 1


def test_check_window_two_bad():
    epsilon = 1.0
    p_arr = np.array([
        [0.0, 0.0],
        [1.0, 10.0],
        [1.0, 20.0],
        [2.0, 0.0],
    ])
    all_good, max_index = check_window(p_arr, epsilon)
    assert not all_good
    assert max_index == 2


def test_check_windows():
    epsilon = 1.0
    p_arr = np.array([
        [0.0, 0.0],
        [1.0, 10.0],
        [1.0, 20.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [5.0, 0.0],
    ])
    windows = [(0, 3), (3, 6)]
    p_arr_ok = np.zeros(len(p_arr)).astype(bool)
    new_windows = check_windows(p_arr, epsilon, windows, p_arr_ok)

    p_arr_ok_gt = np.zeros(len(p_arr)).astype(bool)
    p_arr_ok_gt[np.array([3, 6])] = True
    new_windows_gt = [(0, 2), (2, 3)]

    np_test.assert_equal(p_arr_ok, p_arr_ok_gt)
    assert new_windows_gt == new_windows
