import numpy as np
import numpy.typing as npt
from numba import njit, prange, float64
from numba.np.extensions import cross2d

_rdp_quick_use_cache_ = True


@njit(float64[:](float64[:], float64[:, :]), parallel=True, cache=_rdp_quick_use_cache_)
def _compute_distance_single_point(p1: npt.NDArray[float], p_array: npt.NDArray[float]) -> npt.NDArray[float]:
    num_points = len(p_array)
    out = np.empty(num_points, dtype=p_array.dtype)
    for i in prange(num_points):
        p = p_array[i, :]
        out[i] = np.linalg.norm(p - p1)
    return out


@njit(float64[:](float64[:], float64[:, :], float64[:], float64), parallel=True, cache=_rdp_quick_use_cache_)
def _compute_distance_2d(p1: npt.NDArray[float], p_array: npt.NDArray[float], delta_start_end: npt.NDArray[float],
                         norm_delta_start_end: float) -> npt.NDArray[float]:
    num_points = len(p_array)
    out = np.empty(num_points, dtype=p_array.dtype)
    for i in prange(num_points):
        p = p_array[i, :]
        out[i] = np.abs(cross2d(delta_start_end, p1 - p)) / norm_delta_start_end
    return out


@njit(float64[:](float64[:], float64[:, :], float64[:], float64), parallel=True, cache=_rdp_quick_use_cache_)
def _compute_distance_nd(p1: npt.NDArray[float], p_array: npt.NDArray[float], delta_start_end: npt.NDArray[float],
                         norm_delta_start_end: float) -> npt.NDArray[float]:
    num_points = len(p_array)
    out = np.empty(num_points, dtype=p_array.dtype)
    for i in prange(num_points):
        p = p_array[i, :]
        out[i] = np.abs(np.linalg.norm(np.cross(delta_start_end, p1 - p)))/ norm_delta_start_end
    return out


@njit(float64[:](float64[:], float64[:], float64[:, :]), parallel=True, cache=_rdp_quick_use_cache_)
def compute_distance(p1: npt.NDArray[float], p2: npt.NDArray[float], p_array: npt.NDArray[float]) -> npt.NDArray[float]:

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
