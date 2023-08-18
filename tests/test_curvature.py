import numpy as np
import numpy.testing as np_test
from rdp_quick.curvature import compute_curvature_2d, compute_curvature_and_build_windows


def test_compute_curvature_d2_flat():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    curvature = compute_curvature_2d(x, y)
    curvature_gt = np.array([0.0, 0.0, 0.0])
    np_test.assert_equal(curvature, curvature_gt)


def test_compute_curvature_d2():
    r = 1.0
    x = np.array([-1.0, 0.0, 1.0])
    y = np.sqrt(r**2 - np.square(x))
    curvature = compute_curvature_2d(x, y)
    curvature_gt = np.array([0.35355, 1.0, 0.35355])
    np_test.assert_allclose(curvature, curvature_gt, 0.0001)


def test_compute_curvature_and_build_windows():
    num_osc = 1
    points_per_osc = 100
    n_pts = num_osc * points_per_osc
    x = np.arange(n_pts) / n_pts * num_osc * 2.0 * np.pi
    y = np.sin(x) * 2
    windows = compute_curvature_and_build_windows(x, y)
    windows_gt = [(0, 25), (25, 75), (75, 99)]
    assert windows_gt == windows