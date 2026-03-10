from __future__ import annotations


def safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


def compute_classification_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    total = tp + tn + fp + fn
    accuracy = safe_div(tp + tn, total)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    return {
        "accuracy": accuracy,
        "fpr": fpr,
        "fnr": fnr,
        "precision": precision,
        "recall": recall,
        "total": total,
    }