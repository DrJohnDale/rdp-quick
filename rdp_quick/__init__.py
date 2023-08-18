import numpy as np
import numpy.typing as npt
from rdp_quick.check_window import check_windows
from rdp_quick.curvature import compute_curvature_and_build_windows
import typing


def rdp_initial_windows(p_array: npt.NDArray[float], epsilon: float,
                        initial_windows: typing.List[typing.Tuple[int, int]]) -> npt.NDArray[float]:
    """
    Computes the new points based starting with the windows given
    :param p_array: the data points
    :param epsilon: the threshold
    :param initial_windows: the initial windows
    :return: The down sampled points
    """
    p_array_ok = np.zeros(p_array.shape[0], dtype=bool)
    windows = initial_windows
    num_windows = len(initial_windows)
    while num_windows > 0:
        windows = check_windows(p_array, epsilon, windows, p_array_ok)
        num_windows = len(windows)
    return p_array[p_array_ok]


def rdp_single_initial_window(p_array: npt.NDArray[float], epsilon: float) -> npt.NDArray[float]:
    """
    Computes the new points based starting with one window over all the points
    :param p_array: the data points
    :param epsilon: the threshold
    :return: The down sampled points
    """
    windows = [(0, len(p_array)-1)]
    return rdp_initial_windows(p_array, epsilon, windows)


def get_initial_windows(num_points: int, num_windows: int,
                        points_per_window: int) -> typing.List[typing.Tuple[int, int]]:
    """
    Computes initial windows
    :param num_points: the total number of points
    :param num_windows: the number of windows
    :param points_per_window: the points per window
    :return: a list of initial windows
    """
    windows = list()
    for i in range(num_windows):
        si = i * points_per_window
        ei = si + points_per_window
        if i == num_windows - 1:
            ei = num_points - 1
        windows.append((si, ei))
    return windows


def rdp_num_windows(p_array: npt.NDArray[float], epsilon: float, num_windows) -> npt.NDArray[float]:
    """
    Computes the new points starting with the given number of windows
    :param p_array: the data points
    :param epsilon: the threshold
    :param num_windows: the initial number of windows
    :return: The down sampled points
    """
    if num_windows <= 1:
        return rdp_single_initial_window(p_array, epsilon)

    points_per_window = np.int(np.round(len(p_array)/num_windows))
    windows = get_initial_windows(len(p_array), num_windows, points_per_window)
    return rdp_initial_windows(p_array, epsilon, windows)


def rdp_points_per_window(p_array: npt.NDArray[float], epsilon: float, points_per_window) -> npt.NDArray[float]:
    """
    Computes the new points starting with windows of length points_per_window
    :param p_array: the data points
    :param epsilon: the threshold
    :param points_per_window: the initial number of points per window
    :return: The down sampled points
    """
    if points_per_window >= len(p_array) - 1:
        return rdp_single_initial_window(p_array, epsilon)

    num_windows = int(np.round(len(p_array)/points_per_window))
    windows = get_initial_windows(len(p_array), num_windows, points_per_window)
    return rdp_initial_windows(p_array, epsilon, windows)


def rdp_windows_from_curvature(p_array: npt.NDArray[float], epsilon: float,
                               gradient_nargs: typing.Union[dict, None] = None,
                               peak_find_nargs: typing.Union[dict, None] = None) -> npt.NDArray[float]:
    """
    Computes the new points by first determining the initial windows using the curvature

    The curvature calculation will only use the first two columns as x and y, other dimensions will be ignored

    :param p_array: the data points
    :param epsilon: the threshold
    :param gradient_nargs: any named arguments to pass to the numpy.gradient function
    :param peak_find_nargs: any named arguments to pass to the scipy.signal.find_peaks function
    :return: The down sampled points
    """
    windows = compute_curvature_and_build_windows(p_array[:, 0], p_array[:, 1],
                                                  gradient_nargs=gradient_nargs,
                                                  peak_find_nargs=peak_find_nargs)
    return rdp_initial_windows(p_array, epsilon, windows)
