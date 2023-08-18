import numpy as np
import numpy.testing as np_test
from rdp_quick import rdp_single_initial_window, get_initial_windows


def test_rdp_straight():
    epsilon = 1.0
    p_arr = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [5.0, 0.0],
    ])
    p_arr_selected = rdp_single_initial_window(p_arr, epsilon)
    p_arr_selected_gt = p_arr[[0, -1]]
    np_test.assert_equal(p_arr_selected, p_arr_selected_gt)


def test_rdp_triangle():
    epsilon = 1.0
    p_arr = np.array([
        [0.0, 0.0],
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 20.0],
        [5.0, 10.0],
        [6.0, 0.0],
    ])
    p_arr_selected = rdp_single_initial_window(p_arr, epsilon)
    p_arr_selected_gt = p_arr[[0, 3, -1]]
    np_test.assert_equal(p_arr_selected, p_arr_selected_gt)


def test_rdp_top_hat():
    epsilon = 1.0
    p_arr = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 30.0],
        [4.0, 30.0],
        [5.0, 30.0],
        [6.0, 0.0],
        [7.0, 0.0],
        [8.0, 0.0],
    ])
    p_arr_selected = rdp_single_initial_window(p_arr, epsilon)
    p_arr_selected_gt = p_arr[[0, 2, 3, 5, 6, -1]]
    np_test.assert_equal(p_arr_selected, p_arr_selected_gt)


def test_get_initial_windows():
    initial_windows = get_initial_windows(30, 3, 10)
    initial_windows_gt = [
        (0, 10),
        (10, 20),
        (20, 29),
    ]
    assert initial_windows == initial_windows_gt
