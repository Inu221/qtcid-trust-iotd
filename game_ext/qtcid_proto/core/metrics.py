from __future__ import annotations

import statistics

from .utils import safe_div


def summarize(results: list, runs: int, initial_energy: float) -> dict:
    tp_mean = statistics.mean(r.tp for r in results)
    tn_mean = statistics.mean(r.tn for r in results)
    fp_mean = statistics.mean(r.fp for r in results)
    fn_mean = statistics.mean(r.fn for r in results)

    total = tp_mean + tn_mean + fp_mean + fn_mean
    accuracy = safe_div(tp_mean + tn_mean, total)
    precision = safe_div(tp_mean, tp_mean + fp_mean)
    recall = safe_div(tp_mean, tp_mean + fn_mean)
    fpr = safe_div(fp_mean, fp_mean + tn_mean)
    fnr = safe_div(fn_mean, fn_mean + tp_mean)

    energy_left_mean = statistics.mean(r.energy_left for r in results)

    return {
        "runs": float(runs),
        "mttf_mean": statistics.mean(r.mttf for r in results),
        "mttf_std": statistics.pstdev(r.mttf for r in results) if len(results) > 1 else 0.0,
        "energy_left_mean": energy_left_mean,
        "energy_spent_mean": initial_energy - energy_left_mean,
        "vote_energy_mean": statistics.mean(r.vote_energy for r in results),
        "audit_energy_mean": statistics.mean(r.audit_energy_total for r in results),
        "accuracy_mean": accuracy,
        "precision_mean": precision,
        "recall_mean": recall,
        "fpr_mean": fpr,
        "fnr_mean": fnr,
        "tp_mean": tp_mean,
        "tn_mean": tn_mean,
        "fp_mean": fp_mean,
        "fn_mean": fn_mean,
        "audits_mean": statistics.mean(r.audits for r in results),
        "byzantine_fail_ratio": safe_div(sum(1 for r in results if r.byzantine_failed), len(results)),
        "energy_fail_ratio": safe_div(sum(1 for r in results if r.energy_failed), len(results)),
        "good_left_mean": statistics.mean(r.good_left for r in results),
        "bad_left_mean": statistics.mean(r.bad_left for r in results),
        "evicted_left_mean": statistics.mean(r.evicted_left for r in results),
    }