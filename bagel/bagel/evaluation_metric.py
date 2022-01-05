from sklearn.metrics import *

import numpy as np


def range_lift_with_delay(array: np.ndarray, label: np.ndarray, delay=None, inplace=False) -> np.ndarray:
    """
    :param delay: maximum acceptable delay
    :param array:
    :param label:
    :param inplace:
    :return: new_array
    """
    assert np.shape(array) == np.shape(label)
    if delay is None:
        delay = len(array)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_array = np.copy(array) if not inplace else array
    pos = 0
    for sp in splits:
        if is_anomaly:
            ptr = min(pos + delay + 1, sp)
            new_array[pos: ptr] = np.max(new_array[pos: ptr])
            new_array[ptr: sp] = np.maximum(new_array[ptr: sp], new_array[pos])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        ptr = min(pos + delay + 1, sp)
        new_array[pos: sp] = np.max(new_array[pos: ptr])
    return new_array


def ignore_missing(*args, missing):
    result = []
    for arr in args:
        _arr = np.copy(arr)
        result.append(_arr[missing != 1])
    return tuple(result)


def best_f1score_threshold(indicators: np.ndarray, labels: np.ndarray, return_fscore: bool=False, return_candidates: bool=False):
    ps, rs, ts = precision_recall_curve(labels, indicators)
    fs = 2 * ps * rs / np.clip(ps + rs, a_min=1e-8, a_max=None)
    threshold_candidates = ts
    f1_scores = fs
    best_threshold = threshold_candidates[np.argmax(f1_scores)]
    ret = [best_threshold]
    if return_fscore:
        ret.append(np.max(f1_scores))
    if return_candidates:
        ret.append((threshold_candidates, f1_scores))
    return tuple(ret)
