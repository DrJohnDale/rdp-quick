import numpy as np
import numpy.typing as npt
from rdp_quick.check_window import check_windows
from rdp_quick.curvature import compute_curvature_and_build_windows
import typing


def rdp_initial_windows(p_array: npt.NDArray[float], epsilon: float,
                        initial_windows: typing.List[typing.Tuple[int, int]]) -> npt.NDArray[float]:
    p_array_ok = np.zeros(p_array.shape[0], dtype=bool)
    windows = initial_windows
    num_windows = len(initial_windows)
    while num_windows > 0:
        windows = check_windows(p_array, epsilon, windows, p_array_ok)
        num_windows = len(windows)
    return p_array[p_array_ok]


def rdp_single_initial_window(p_array: npt.NDArray[float], epsilon: float) -> npt.NDArray[float]:
    windows = [(0, len(p_array)-1)]
    return rdp_initial_windows(p_array, epsilon, windows)


def get_initial_windows(num_points: int, num_windows: int,
                        points_per_window: int) -> typing.List[typing.Tuple[int, int]]:
    windows = list()
    for i in range(num_windows):
        si = i * points_per_window
        ei = si + points_per_window
        if i == num_windows - 1:
            ei = num_points - 1
        windows.append((si, ei))
    return windows


def rdp_num_windows(p_array: npt.NDArray[float], epsilon: float, num_windows) -> npt.NDArray[float]:
    if num_windows <= 1:
        return rdp_single_initial_window(p_array, epsilon)

    points_per_window = np.int(np.round(len(p_array)/num_windows))
    windows = get_initial_windows(len(p_array), num_windows, points_per_window)
    return rdp_initial_windows(p_array, epsilon, windows)


def rdp_points_per_window(p_array: npt.NDArray[float], epsilon: float, points_per_window) -> npt.NDArray[float]:
    if points_per_window >= len(p_array) - 1:
        return rdp_single_initial_window(p_array, epsilon)

    num_windows = int(np.round(len(p_array)/points_per_window))
    windows = get_initial_windows(len(p_array), num_windows, points_per_window)
    return rdp_initial_windows(p_array, epsilon, windows)


def rdp_windows_from_curvature(p_array: npt.NDArray[float], epsilon: float,
                               gradient_nargs: typing.Union[dict, None] = None,
                               peak_find_nargs: typing.Union[dict, None] = None) -> npt.NDArray[float]:
    windows = compute_curvature_and_build_windows(p_array[:, 0], p_array[:, 1],
                                                  gradient_nargs=gradient_nargs,
                                                  peak_find_nargs=peak_find_nargs)
    return rdp_initial_windows(p_array, epsilon, windows)
