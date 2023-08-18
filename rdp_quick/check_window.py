import numpy as np
import numpy.typing as npt
from numba import njit, float64, types, boolean, int64
import typing
from rdp_quick.compute_distance import compute_distance, _rdp_quick_use_cache_


@njit(types.Tuple((boolean, int64))(float64[:, :], float64), parallel=True, cache=_rdp_quick_use_cache_)
def check_window(p_array: npt.NDArray[float], epsilon: float) -> typing.Tuple[bool, int]:
    """
    Check if the given window.  The line is made from the first and last point.
    :param p_array: the points to check
    :param epsilon: the threshold to test against
    :return: if any distance is greater than epsilon False, max_index, if all withing epsilon then False, 0
    """
    if p_array.shape[0] <= 2:
        return True, 0
    dists = compute_distance(p_array[0, :], p_array[-1, :], p_array[1:-1, :])
    arg_max = int(np.argmax(dists))
    val_max = dists[arg_max]
    if val_max > epsilon:
        return False, arg_max + 1
    else:
        return True, 0


def check_windows(p_array: npt.NDArray[float], epsilon: float, windows: typing.List[typing.Tuple[int, int]], p_array_ok: npt.NDArray[bool]):
    """
    test all the current windows, if any do not pass the test then create the new windows
    :param p_array: The data points
    :param epsilon: The threshold
    :param windows: the initial windows
    :param p_array_ok: Which points are accepted
    :return: a list of new windows
    """
    new_windows = list()
    for win in windows:
        start, end = win
        ok, arg_max = check_window(p_array[start:end+1, :], epsilon)
        if ok:
            p_array_ok[win[0]] = True
            p_array_ok[win[1]] = True
        else:
            new_windows.append((win[0], win[0] + arg_max))
            new_windows.append((win[0] + arg_max, win[1]))
    return new_windows
