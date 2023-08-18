import numpy as np
import numpy.typing as npt
import typing
from scipy.signal import find_peaks


def compute_curvature_2d(x: npt.NDArray[float], y: npt.NDArray[float], gradient_nargs: typing.Union[dict, None] = None) -> npt.NDArray[float]:
    gradient_nargs = gradient_nargs if gradient_nargs is not None else dict()
    diff_x = np.gradient(x, **gradient_nargs)
    diff_y = np.gradient(y, **gradient_nargs)
    diff_diff_x = np.gradient(diff_x, **gradient_nargs)
    diff_diff_y = np.gradient(diff_y, **gradient_nargs)
    curvature = np.abs(diff_diff_x * diff_y - diff_x * diff_diff_y) / (diff_x * diff_x + diff_y * diff_y)**1.5
    return curvature


def compute_curvature_and_build_windows(x: npt.NDArray[float], y: npt.NDArray[float],
                                        gradient_nargs: typing.Union[dict, None] = None,
                                        peak_find_nargs: typing.Union[dict, None] = None) -> typing.List[typing.Tuple[int, int]]:
    peak_find_nargs = peak_find_nargs if peak_find_nargs is not None else dict()
    curvature = compute_curvature_2d(x, y, gradient_nargs=gradient_nargs)
    peaks = np.array(find_peaks(curvature, **peak_find_nargs)[0])
    points = np.append(0, peaks)
    points = np.append(points, len(curvature) - 1)
    windows = list()
    for pi, p_val in enumerate(points[0:-1]):
        windows.append((p_val, points[pi + 1]))
    return windows
