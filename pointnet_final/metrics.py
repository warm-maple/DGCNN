from __future__ import annotations

import numpy as np


def instance_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    mask = labels >= 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(preds[mask] == labels[mask]))


def class_accuracy(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    accs: list[float] = []
    for cls in range(num_classes):
        mask = labels == cls
        if np.any(mask):
            accs.append(float(np.mean(preds[mask] == labels[mask])))
    return float(np.mean(accs)) if accs else float("nan")


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(preds, labels):
        if label >= 0:
            mat[int(label), int(pred)] += 1
    return mat

