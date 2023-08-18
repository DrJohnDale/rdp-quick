import numpy as np
import numpy.typing as npt
from numba import njit, prange, float64
from numba.np.extensions import cross2d

_rdp_quick_use_cache_ = True


@njit(float64[:](float64[:], float64[:, :]), parallel=True, cache=_rdp_quick_use_cache_)
def _compute_distance_single_point(p1: npt.NDArray[float], p_array: npt.NDArray[float]) -> npt.NDArray[float]:
    """
    compute distance from a single point
    :param p1: the point to compute the distance from
    :param p_array: the points to compute the distance to
    :return: the distances
    """
    num_points = len(p_array)
    out = np.empty(num_points, dtype=p_array.dtype)
    for i in prange(num_points):
        p = p_array[i, :]
        out[i] = np.linalg.norm(p - p1)
    return out


@njit(float64[:](float64[:], float64[:, :], float64[:], float64), parallel=True, cache=_rdp_quick_use_cache_)
def _compute_distance_2d(p1: npt.NDArray[float], p_array: npt.NDArray[float], delta_start_end: npt.NDArray[float],
                         norm_delta_start_end: float) -> npt.NDArray[float]:
    """
    Compute the distance to the line with 2d points
    :param p1: the initial point
    :param p_array: the point to compute the distance to
    :param delta_start_end: the vector difference between the start and the end point
    :param norm_delta_start_end: the normal of the vector between the start and end point
    :return: an array of distances
    """
    num_points = len(p_array)
    out = np.empty(num_points, dtype=p_array.dtype)
    for i in prange(num_points):
        p = p_array[i, :]
        out[i] = np.abs(cross2d(delta_start_end, p1 - p)) / norm_delta_start_end
    return out


@njit(float64[:](float64[:], float64[:, :], float64[:], float64), parallel=True, cache=_rdp_quick_use_cache_)
def _compute_distance_nd(p1: npt.NDArray[float], p_array: npt.NDArray[float], delta_start_end: npt.NDArray[float],
                         norm_delta_start_end: float) -> npt.NDArray[float]:
    """
    Compute the distance to the line with nd points
    :param p1: the initial point
    :param p_array: the point to compute the distance to
    :param delta_start_end: the vector difference between the start and the end point
    :param norm_delta_start_end: the normal of the vector between the start and end point
    :return: an array of distances
    """
    num_points = len(p_array)
    out = np.empty(num_points, dtype=p_array.dtype)
    for i in prange(num_points):
        p = p_array[i, :]
        out[i] = np.abs(np.linalg.norm(np.cross(delta_start_end, p1 - p))) / norm_delta_start_end
    return out


@njit(float64[:](float64[:], float64[:], float64[:, :]), parallel=True, cache=_rdp_quick_use_cache_)
def compute_distance(p1: npt.NDArray[float], p2: npt.NDArray[float], p_array: npt.NDArray[float]) -> npt.NDArray[float]:
    """
    Compute the distances to the points from a line made from p1 and p1
    :param p1: The start point
    :param p2: The end point
    :param p_array: The points to compute the distance to
    :return: an array of distances
    """
    p1_equal_p2 = np.all(np.equal(p1, p2))
    if p1_equal_p2:
        return _compute_distance_single_point(p1, p_array)
    else:
        delta_start_end = p2 - p1
        norm_delta_start_end = np.linalg.norm(delta_start_end)
        if p_array.shape[1] == 2:
            return _compute_distance_2d(p1, p_array, delta_start_end, norm_delta_start_end)
        else:
            return _compute_distance_nd(p1, p_array, delta_start_end, norm_delta_start_end)
