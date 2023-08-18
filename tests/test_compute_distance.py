import numpy as np
import numpy.testing as np_test
from rdp_quick.compute_distance import compute_distance


def test_single_point():
    p1 = np.array([0.0, 0.0])
    p2 = p1
    p_arr = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    distances = compute_distance(p1, p2, p_arr)
    distances_truth = np.array([0.0, 1.0, 1.0, np.sqrt(2.0)])
    np_test.assert_equal(distances, distances_truth)


def test_2d():
    p1 = np.array([0.0, 0.0])
    p2 = np.array([2.0, 0.0])
    p_arr = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [1.0, 1.0],
        [4.0, 1.0]
    ])
    distances = compute_distance(p1, p2, p_arr)
    distances_truth = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    np_test.assert_equal(distances, distances_truth)


def test_3d():
    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([2.0, 0.0, 1.0])
    p_arr = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [3.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [4.0, 1.0, 1.0]
    ])
    distances = compute_distance(p1, p2, p_arr)
    distances_truth = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    np_test.assert_equal(distances, distances_truth)
