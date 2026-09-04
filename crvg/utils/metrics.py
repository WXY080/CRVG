"""Evaluation metrics for REC (Referring Expression Comprehension)."""
import numpy as np


def acc_at(ious, threshold):
    """Accuracy at a given IoU threshold."""
    return 100 * sum(1 for i in ious if i >= threshold) / max(1, len(ious))


def acc50(ious):
    return acc_at(ious, 0.5)


def acc75(ious):
    return acc_at(ious, 0.75)


def acc90(ious):
    return acc_at(ious, 0.9)


def mean_iou(ious):
    return 100 * float(np.mean(ious)) if ious else 0.0


def acc_miou(ious):
    """Internal metrics are fractions; formatted tables convert to percentages."""
    values = list(ious)
    if any(v is None or not np.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Metrics require finite ground-truth IoUs in [0,1]")
    return {"acc0.5": acc50(values) / 100, "acc0.75": acc75(values) / 100,
            "acc0.9": acc90(values) / 100, "miou": mean_iou(values) / 100}
