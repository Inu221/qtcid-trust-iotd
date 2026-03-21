from __future__ import annotations

from pathlib import Path
import csv
import statistics

import matplotlib.pyplot as plt
import numpy as np

from game_ext.qtcid_repro.qtcid_core import (
    QTCIDConfig,
    QTCIDSimulator,
)
from game_ext.qtcid_repro.ta_qtcid_core import (
    TAQTCIDConfig,
    TAQTCIDSimulator,
)


PA_VALUE = 1.0
TIDS_VALUE = 600
M_VALUES = [1, 3, 5, 7, 9]

RUNS = 80
SEED = 42
MAX_TIME = 16000

OUT_DIR = Path("results/rinc_m_voters_study")

METHOD_COLORS = {
    "Q-TCID": "#2ca02c",
    "TA-QTCID": "#d62728",
}


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def safe_pct_improvement(old: float, new: float) -> float:
    if old == 0:
        return 0.0
    return 100.0 * (old - new) / old


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_qtcid_cfg(m_voters: int, runs: int = RUNS) -> QTCIDConfig:
    return QTCIDConfig(
        n_nodes=128,
        n_neighbors=32,
        m_voters=m_voters,
        tids=TIDS_VALUE,
        lambda_capture=1 / 3600,
        hpfn=0.05,
        hpfp=0.05,
        pa=PA_VALUE,
        pc=0.5,
        beta=1.0,
        runs=runs,
        max_time=MAX_TIME,
        seed=SEED,
    )


def build_taqtcid_cfg(m_voters: int, runs: int = RUNS) -> TAQTCIDConfig:
    return TAQTCIDConfig(
        n_nodes=128,
        n_neighbors=32,
        m_voters=m_voters,
        tids=TIDS_VALUE,
        lambda_capture=1 / 3600,
        hpfn=0.05,
        hpfp=0.05,
        pa=PA_VALUE,
        pc=0.5,
        beta=1.0,
        runs=runs,
        max_time=MAX_TIME,
        seed=SEED,
        trust_mode="adaptive",
    )


def run_family_manual(cfg, simulator_cls):
    results = []

    for i in range(cfg.runs):
        sim = simulator_cls(cfg, seed=cfg.seed + i)
        results.append(sim.run())

    tp_vals = [r.tp for r in results]
    tn_vals = [r.tn for r in results]
    fp_vals = [r.fp for r in results]
    fn_vals = [r.fn for r in results]

    mttf_vals = [r.mttf for r in results]
    e_total_vals = [(cfg.initial_system_energy - r.energy_left) for r in results]

    cmvi_vals = [r.cmvi for r in results]
    false_good_vals = [r.false_good_evictions for r in results]
    bad_retained_vals = [r.bad_nodes_retained for r in results]
    mismatch_vals = [r.audit_mismatch_events for r in results]

    tp_mean = mean(tp_vals)
    tn_mean = mean(tn_vals)
    fp_mean = mean(fp_vals)
    fn_mean = mean(fn_vals)

    total = tp_mean + tn_mean + fp_mean + fn_mean
    accuracy = (tp_mean + tn_mean) / total if total > 0 else 0.0
    precision = tp_mean / (tp_mean + fp_mean) if (tp_mean + fp_mean) > 0 else 0.0
    recall = tp_mean / (tp_mean + fn_mean) if (tp_mean + fn_mean) > 0 else 0.0

    return {
        "runs": cfg.runs,
        "mttf_mean": mean(mttf_vals),
        "mttf_std": std(mttf_vals),
        "accuracy_mean": accuracy,
        "precision_mean": precision,
        "recall_mean": recall,
        "energy_spent_mean": mean(e_total_vals),
        "cmvi_mean": mean(cmvi_vals),
        "cmvi_std": std(cmvi_vals),
        "false_good_evictions_mean": mean(false_good_vals),
        "bad_nodes_retained_mean": mean(bad_retained_vals),
        "audit_mismatch_events_mean": mean(mismatch_vals),
    }


def plot_accuracy(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    x = [int(r["m_voters"]) for r in rows if r["method"] == "Q-TCID"]
    q_acc = [float(r["accuracy_mean"]) for r in rows if r["method"] == "Q-TCID"]
    ta_acc = [float(r["accuracy_mean"]) for r in rows if r["method"] == "TA-QTCID"]

    ax.plot(x, q_acc, marker="o", linewidth=2.2, label="Q-TCID")
    ax.plot(x, ta_acc, marker="o", linewidth=2.2, label="TA-QTCID")

    ax.set_xlabel("Число голосующих соседей m")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_accuracy_vs_m.png", dpi=260)
    plt.close()


def plot_cmvi(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    x = [int(r["m_voters"]) for r in rows if r["method"] == "Q-TCID"]
    q_cmvi = [float(r["cmvi_mean"]) for r in rows if r["method"] == "Q-TCID"]
    q_std = [float(r["cmvi_std"]) for r in rows if r["method"] == "Q-TCID"]

    ta_cmvi = [float(r["cmvi_mean"]) for r in rows if r["method"] == "TA-QTCID"]
    ta_std = [float(r["cmvi_std"]) for r in rows if r["method"] == "TA-QTCID"]

    ax.errorbar(x, q_cmvi, yerr=q_std, marker="o", linewidth=2.2, capsize=4, label="Q-TCID")
    ax.errorbar(x, ta_cmvi, yerr=ta_std, marker="o", linewidth=2.2, capsize=4, label="TA-QTCID")

    ax.set_xlabel("Число голосующих соседей m")
    ax.set_ylabel("CMVI, усл. ед.")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_cmvi_vs_m.png", dpi=260)
    plt.close()


def plot_cmvi_bars_by_m(rows: list[dict]) -> None:
    """
    Компактный набор столбчатых диаграмм: для каждого m отдельный подграфик
    с абсолютными значениями CMVI у методов Q-TCID и TA-QTCID.
    """
    m_vals = sorted([int(r["m_voters"]) for r in rows if r["method"] == "Q-TCID"])

    fig = plt.figure(figsize=(13.2, 7.2))
    grid = fig.add_gridspec(2, 6)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]

    for ax, m in zip(axes, m_vals):
        q = next(r for r in rows if r["method"] == "Q-TCID" and int(r["m_voters"]) == m)
        ta = next(r for r in rows if r["method"] == "TA-QTCID" and int(r["m_voters"]) == m)

        labels = ["Q-TCID", "TA-QTCID"]
        values = [float(q["cmvi_mean"]), float(ta["cmvi_mean"])]
        colors = [METHOD_COLORS[label] for label in labels]

        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.0)

        v_min = min(values)
        v_max = max(values)
        span = max(v_max - v_min, v_max * 0.08, 0.4)
        lower = max(0.0, v_min - 0.35 * span)
        upper = v_max + 0.45 * span
        ax.set_ylim(lower, upper)

        ax.set_title(rf"$m = {m}$")
        ax.set_ylabel("CMVI, усл. ед.")
        ax.grid(True, axis="y", alpha=0.25)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.04 * span,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.suptitle("Абсолютные значения CMVI для Q-TCID и TA-QTCID при разных m", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_DIR / "fig_cmvi_absolute_bars_by_m.png", dpi=260)
    plt.close()


def plot_components(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    x = [int(r["m_voters"]) for r in rows if r["method"] == "Q-TCID"]

    metrics = [
        ("false_good_evictions_mean", "Ошибочно исключенные\nкорректные узлы, ср."),
        ("bad_nodes_retained_mean", "Сохраненные вредоносные\nузлы, ср."),
        ("audit_mismatch_events_mean", "Рассогласования\nvote/audit, ср."),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        q_vals = [float(next(r for r in rows if r["method"] == "Q-TCID" and int(r["m_voters"]) == m)[metric]) for m in x]
        ta_vals = [float(next(r for r in rows if r["method"] == "TA-QTCID" and int(r["m_voters"]) == m)[metric]) for m in x]

        ax.plot(x, q_vals, marker="o", linewidth=2.2, label="Q-TCID")
        ax.plot(x, ta_vals, marker="o", linewidth=2.2, label="TA-QTCID")
        ax.set_xlabel("Число голосующих соседей m")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03),
               ncol=2, fontsize=11, frameon=True)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(OUT_DIR / "fig_cmvi_components_vs_m.png", dpi=260)
    plt.close()

def plot_cmvi_component_improvement(rows: list[dict]) -> None:
    """
    Рисунок 2.
    Относительное изменение компонент CMVI при переходе
    от Q-TCID к TA-QTCID для каждого значения m.
    """
    fig, ax = plt.subplots(figsize=(10.8, 5.6))

    x_vals = sorted([int(r["m_voters"]) for r in rows if r["method"] == "Q-TCID"])

    delta_false_good = []
    delta_bad_retained = []
    delta_mismatch = []

    for m in x_vals:
        q = next(r for r in rows if r["method"] == "Q-TCID" and int(r["m_voters"]) == m)
        ta = next(r for r in rows if r["method"] == "TA-QTCID" and int(r["m_voters"]) == m)

        q_fg = float(q["false_good_evictions_mean"])
        ta_fg = float(ta["false_good_evictions_mean"])

        q_br = float(q["bad_nodes_retained_mean"])
        ta_br = float(ta["bad_nodes_retained_mean"])

        q_mm = float(q["audit_mismatch_events_mean"])
        ta_mm = float(ta["audit_mismatch_events_mean"])

        delta_false_good.append(safe_pct_improvement(q_fg, ta_fg))
        delta_bad_retained.append(safe_pct_improvement(q_br, ta_br))
        delta_mismatch.append(safe_pct_improvement(q_mm, ta_mm))

    x = np.arange(len(x_vals))
    width = 0.24

    ax.bar(
        x - width,
        delta_false_good,
        width=width,
        edgecolor="black",
        label="Ошибочные исключения корректных узлов",
    )
    ax.bar(
        x,
        delta_bad_retained,
        width=width,
        edgecolor="black",
        label="Сохраненные вредоносные узлы",
    )
    ax.bar(
        x + width,
        delta_mismatch,
        width=width,
        edgecolor="black",
        label="Рассогласования vote/audit",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x_vals])
    ax.set_xlabel("Число голосующих соседей m")
    ax.set_ylabel("Относительное снижение, %")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_cmvi_component_improvement_vs_m.png", dpi=260)
    plt.close()


def plot_cmvi_dumbbell(rows: list[dict]) -> None:
    """
    Рисунок 3.
    Dumbbell-plot по CMVI для Q-TCID и TA-QTCID.
    """
    fig, ax = plt.subplots(figsize=(9.6, 5.8))

    m_vals = sorted([int(r["m_voters"]) for r in rows if r["method"] == "Q-TCID"])
    y = np.arange(len(m_vals))

    q_vals = []
    ta_vals = []

    for m in m_vals:
        q = next(r for r in rows if r["method"] == "Q-TCID" and int(r["m_voters"]) == m)
        ta = next(r for r in rows if r["method"] == "TA-QTCID" and int(r["m_voters"]) == m)

        q_vals.append(float(q["cmvi_mean"]))
        ta_vals.append(float(ta["cmvi_mean"]))

    # соединительные линии
    for i in range(len(m_vals)):
        ax.plot(
            [q_vals[i], ta_vals[i]],
            [y[i], y[i]],
            linewidth=2.0,
            color="gray",
            alpha=0.8,
            zorder=1,
        )

    # точки методов
    ax.scatter(q_vals, y, s=80, label="Q-TCID", zorder=3)
    ax.scatter(ta_vals, y, s=80, label="TA-QTCID", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([f"m = {v}" for v in m_vals])
    ax.set_xlabel("CMVI, усл. ед.")
    ax.set_ylabel("Число голосующих соседей")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend()

    # подписи значений возле точек
    x_span = max(max(q_vals), max(ta_vals)) - min(min(q_vals), min(ta_vals))
    dx = 0.015 * x_span if x_span > 0 else 0.2

    for i in range(len(m_vals)):
        ax.text(q_vals[i] + dx, y[i] + 0.06, f"{q_vals[i]:.2f}", fontsize=9)
        ax.text(ta_vals[i] + dx, y[i] - 0.16, f"{ta_vals[i]:.2f}", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_cmvi_dumbbell_vs_m.png", dpi=260)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    compact_rows = []

    for m in M_VALUES:
        print(f"Running: Pa={PA_VALUE:.2f}, TIDS={TIDS_VALUE}, m={m}")

        q_cfg = build_qtcid_cfg(m)
        ta_cfg = build_taqtcid_cfg(m)

        q = run_family_manual(q_cfg, QTCIDSimulator)
        ta = run_family_manual(ta_cfg, TAQTCIDSimulator)

        imp = safe_pct_improvement(q["cmvi_mean"], ta["cmvi_mean"])

        rows.append({
            "pa": PA_VALUE,
            "tids": TIDS_VALUE,
            "m_voters": m,
            "method": "Q-TCID",
            "runs": q["runs"],
            "mttf_mean": round(q["mttf_mean"], 4),
            "mttf_std": round(q["mttf_std"], 4),
            "accuracy_mean": round(q["accuracy_mean"], 6),
            "precision_mean": round(q["precision_mean"], 6),
            "recall_mean": round(q["recall_mean"], 6),
            "energy_spent_mean": round(q["energy_spent_mean"], 4),
            "cmvi_mean": round(q["cmvi_mean"], 4),
            "cmvi_std": round(q["cmvi_std"], 4),
            "false_good_evictions_mean": round(q["false_good_evictions_mean"], 4),
            "bad_nodes_retained_mean": round(q["bad_nodes_retained_mean"], 4),
            "audit_mismatch_events_mean": round(q["audit_mismatch_events_mean"], 4),
            "cmvi_improvement_vs_q_pct": 0.0,
        })

        rows.append({
            "pa": PA_VALUE,
            "tids": TIDS_VALUE,
            "m_voters": m,
            "method": "TA-QTCID",
            "runs": ta["runs"],
            "mttf_mean": round(ta["mttf_mean"], 4),
            "mttf_std": round(ta["mttf_std"], 4),
            "accuracy_mean": round(ta["accuracy_mean"], 6),
            "precision_mean": round(ta["precision_mean"], 6),
            "recall_mean": round(ta["recall_mean"], 6),
            "energy_spent_mean": round(ta["energy_spent_mean"], 4),
            "cmvi_mean": round(ta["cmvi_mean"], 4),
            "cmvi_std": round(ta["cmvi_std"], 4),
            "false_good_evictions_mean": round(ta["false_good_evictions_mean"], 4),
            "bad_nodes_retained_mean": round(ta["bad_nodes_retained_mean"], 4),
            "audit_mismatch_events_mean": round(ta["audit_mismatch_events_mean"], 4),
            "cmvi_improvement_vs_q_pct": round(imp, 4),
        })

        compact_rows.append({
            "m_voters": m,
            "q_accuracy_mean": round(q["accuracy_mean"], 6),
            "ta_accuracy_mean": round(ta["accuracy_mean"], 6),
            "q_mttf_mean": round(q["mttf_mean"], 4),
            "ta_mttf_mean": round(ta["mttf_mean"], 4),
            "q_energy_spent_mean": round(q["energy_spent_mean"], 4),
            "ta_energy_spent_mean": round(ta["energy_spent_mean"], 4),
            "q_cmvi_mean": round(q["cmvi_mean"], 4),
            "ta_cmvi_mean": round(ta["cmvi_mean"], 4),
            "cmvi_improvement_pct": round(imp, 4),
        })

    write_csv(OUT_DIR / "table_m_voters_detailed.csv", rows)
    write_csv(OUT_DIR / "table_m_voters_compact.csv", compact_rows)

    plot_cmvi_bars_by_m(rows)
    plot_cmvi_component_improvement(rows)
    plot_cmvi_dumbbell(rows)

    print("\nSaved:")
    print(OUT_DIR / "table_m_voters_detailed.csv")
    print(OUT_DIR / "table_m_voters_compact.csv")
    print(OUT_DIR / "fig_cmvi_absolute_bars_by_m.png")
    print(OUT_DIR / "fig_cmvi_component_improvement_vs_m.png")
    print(OUT_DIR / "fig_cmvi_dumbbell_vs_m.png")

if __name__ == "__main__":
    main()