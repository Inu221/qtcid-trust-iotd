from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import csv
import statistics

import matplotlib.pyplot as plt
import numpy as np

from game_ext.qtcid_repro.wang.bvs_core import (
    WangBVSConfig,
    run_wang_bvs_monte_carlo,
)
from game_ext.qtcid_repro.qtcid_core import (
    QTCIDConfig,
    QTCIDSimulator,
)
from game_ext.qtcid_repro.ta_qtcid_core import (
    TAQTCIDConfig,
    TAQTCIDSimulator,
)


PA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
TIDS_VALUES = [50, 100, 200, 350, 600, 1000, 1500]

# Для компактных таблиц в статье
REPRESENTATIVE_TIDS = [200, 600, 1000]

RUNS = 80
SEED = 42
MAX_TIME = 16000

OUT_DIR = Path("results/final_qtcid_taqtcid_study")


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def safe_pct_improvement(old: float, new: float) -> float:
    if old == 0:
        return 0.0
    return 100.0 * (old - new) / old


def build_bvs_cfg(pa: float, tids: int, runs: int = RUNS) -> WangBVSConfig:
    return WangBVSConfig(
        n_nodes=128,
        n_neighbors=32,
        m_voters=5,
        tids=tids,
        lambda_capture=1 / 3600,
        hpfn=0.05,
        hpfp=0.05,
        pa=pa,
        pc=0.5,
        beta=1.0,
        runs=runs,
        max_time=MAX_TIME,
        seed=SEED,
    )


def build_qtcid_cfg(pa: float, tids: int, runs: int = RUNS) -> QTCIDConfig:
    return QTCIDConfig(
        n_nodes=128,
        n_neighbors=32,
        m_voters=5,
        tids=tids,
        lambda_capture=1 / 3600,
        hpfn=0.05,
        hpfp=0.05,
        pa=pa,
        pc=0.5,
        beta=1.0,
        runs=runs,
        max_time=MAX_TIME,
        seed=SEED,
    )


def build_taqtcid_cfg(pa: float, tids: int, runs: int = RUNS) -> TAQTCIDConfig:
    return TAQTCIDConfig(
        n_nodes=128,
        n_neighbors=32,
        m_voters=5,
        tids=tids,
        lambda_capture=1 / 3600,
        hpfn=0.05,
        hpfp=0.05,
        pa=pa,
        pc=0.5,
        beta=1.0,
        runs=runs,
        max_time=MAX_TIME,
        seed=SEED,
    )


def run_qtcid_family_manual(cfg, simulator_cls):
    """
    Возвращает агрегированную сводку + CMVI, потому что SummaryStats этого не хранит.
    """
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
    e_vote_vals = [r.energy_spent_voting for r in results]
    e_audit_vals = [r.energy_spent_audit for r in results]
    audits_vals = [r.audits for r in results]

    good_vals = [r.good_left for r in results]
    bad_vals = [r.bad_left for r in results]
    evicted_vals = [r.evicted_left for r in results]

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
    fpr = fp_mean / (fp_mean + tn_mean) if (fp_mean + tn_mean) > 0 else 0.0
    fnr = fn_mean / (fn_mean + tp_mean) if (fn_mean + tp_mean) > 0 else 0.0

    return {
        "runs": cfg.runs,
        "mttf_mean": mean(mttf_vals),
        "mttf_std": std(mttf_vals),
        "accuracy_mean": accuracy,
        "precision_mean": precision,
        "recall_mean": recall,
        "fpr_mean": fpr,
        "fnr_mean": fnr,
        "energy_spent_mean": mean(e_total_vals),
        "energy_voting_mean": mean(e_vote_vals),
        "energy_audit_mean": mean(e_audit_vals),
        "audits_mean": mean(audits_vals),
        "good_left_mean": mean(good_vals),
        "bad_left_mean": mean(bad_vals),
        "evicted_left_mean": mean(evicted_vals),
        "cmvi_mean": mean(cmvi_vals),
        "cmvi_std": std(cmvi_vals),
        "false_good_evictions_mean": mean(false_good_vals),
        "bad_nodes_retained_mean": mean(bad_retained_vals),
        "audit_mismatch_events_mean": mean(mismatch_vals),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


_PANEL_TITLES = {
    0.0:  r"(а) $P_a=0$",
    0.25: r"(б) $P_a=0.25$",
    0.5:  r"(в) $P_a=0.5$",
    0.75: r"(г) $P_a=0.75$",
    1.0:  r"(д) $P_a=1$",
}

_COLORS = {
    "Q-TCID": "#2ca02c",
    "TA-QTCID": "#d62728",
}


def _five_panel_fig(figsize=(16.5, 10.0)):
    """Фигура из 5 подграфиков: 3 сверху, 2 по центру снизу."""
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=figsize, layout="constrained")
    # резервируем нижние 8% под общую легенду
    fig.get_layout_engine().set(rect=(0, 0.08, 1, 1))
    gs = GridSpec(2, 6, figure=fig)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]
    return fig, axes


def plot_mttf_by_pa(detail_rows: list[dict]) -> None:
    fig, axes = _five_panel_fig()

    for idx, pa in enumerate(PA_VALUES):
        ax = axes[idx]
        rows = [r for r in detail_rows if float(r["pa"]) == pa]

        x = [int(r["tids"]) for r in rows]
        y_q = [float(r["q_mttf_mean"]) for r in rows]
        y_ta = [float(r["ta_mttf_mean"]) for r in rows]

        ax.plot(x, y_q, marker="o", linewidth=2.2, color=_COLORS["Q-TCID"], label="Q-TCID")
        ax.plot(x, y_ta, marker="o", linewidth=2.2, color=_COLORS["TA-QTCID"], label="TA-QTCID")

        ax.set_title(_PANEL_TITLES[pa], fontsize=13)
        ax.set_xlabel(r"Интервал диагностики $T_{IDS}$, с")
        ax.set_ylabel("Среднее время до отказа (MTTF), с")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, fontsize=12, frameon=True)
    plt.savefig(OUT_DIR / "fig_mttf_qtcid_vs_taqtcid.png", dpi=260, bbox_inches="tight")
    plt.close()


def plot_cmvi_by_pa(detail_rows: list[dict]) -> None:
    fig, axes = _five_panel_fig()

    for idx, pa in enumerate(PA_VALUES):
        ax = axes[idx]
        rows = [r for r in detail_rows if float(r["pa"]) == pa]

        x = [int(r["tids"]) for r in rows]
        y_q = [float(r["q_cmvi_mean"]) for r in rows]
        y_ta = [float(r["ta_cmvi_mean"]) for r in rows]

        ax.plot(x, y_q, marker="o", linewidth=2.2, color=_COLORS["Q-TCID"], label="Q-TCID")
        ax.plot(x, y_ta, marker="o", linewidth=2.2, color=_COLORS["TA-QTCID"], label="TA-QTCID")

        ax.set_title(_PANEL_TITLES[pa], fontsize=13)
        ax.set_xlabel(r"Интервал диагностики $T_{IDS}$, с")
        ax.set_ylabel("Уязвимость (CMVI), усл. ед.")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, fontsize=12, frameon=True)
    plt.savefig(OUT_DIR / "fig_cmvi_qtcid_vs_taqtcid.png", dpi=260, bbox_inches="tight")
    plt.close()


def plot_cmvi_improvement(detail_rows: list[dict]) -> None:
    fig, axes = _five_panel_fig()

    for idx, pa in enumerate(PA_VALUES):
        ax = axes[idx]
        rows = [r for r in detail_rows if float(r["pa"]) == pa]

        x = [int(r["tids"]) for r in rows]
        y = [float(r["cmvi_improvement_pct"]) for r in rows]

        ax.plot(x, y, marker="o", linewidth=2.2, color="#1f77b4")
        ax.set_title(_PANEL_TITLES[pa], fontsize=13)
        ax.set_xlabel(r"Интервал диагностики $T_{IDS}$, с")
        ax.set_ylabel("Улучшение CMVI, %")
        ax.grid(True, alpha=0.25)

    plt.savefig(OUT_DIR / "fig_cmvi_improvement_pct.png", dpi=260, bbox_inches="tight")
    plt.close()


def plot_representative_bars(rep_rows: list[dict]) -> None:
    """
    Компактный график для статьи: средний процентный выигрыш
    TA-QTCID над Q-TCID по CMVI для репрезентативных TIDS.
    """
    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    labels = []
    values = []

    for tids in REPRESENTATIVE_TIDS:
        subset = [r for r in rep_rows if int(r["tids"]) == tids]
        labels.append(f"$T_{{IDS}}={tids}$")
        values.append(mean([float(r["cmvi_improvement_pct"]) for r in subset]))

    ax.bar(labels, values, edgecolor="black")
    ax.set_xlabel(r"Интервал диагностики $T_{IDS}$, с")
    ax.set_ylabel("Среднее улучшение CMVI, %")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_avg_cmvi_improvement_representative.png", dpi=260)
    plt.close()


def format_word_tables(detail_rows: list[dict], rep_rows: list[dict]) -> None:
    """Генерирует HTML-таблицы для вставки в Word (копировать из браузера)."""

    _CSS = """<style>
body { font-family: Arial, sans-serif; font-size: 11pt; }
h3 { margin-top: 2em; }
table { border-collapse: collapse; margin-bottom: 2em; }
th, td { border: 1px solid #555; padding: 4px 10px; text-align: center; }
th { background: #d0d8e8; }
th.g2 { background: #a8b8d0; }
tr:nth-child(even) td { background: #f5f5f5; }
td.pa { font-weight: bold; background: #e8eef8; }
td.better { color: #155215; font-weight: bold; }
</style>"""

    out_lines = [f"<!DOCTYPE html>\n<html lang='ru'>\n<head>\n<meta charset='utf-8'>\n{_CSS}\n</head>\n<body>"]

    # ── Таблица 1: репрезентативная Pa × TIDS ───────────────────────────────
    out_lines.append("<h3>Таблица 1. Сравнение Q-TCID и TA-QTCID по ключевым метрикам</h3>")
    out_lines.append("<table>")
    out_lines.append(
        "<tr>"
        "<th rowspan='2'>P<sub>a</sub></th>"
        "<th rowspan='2'>T<sub>IDS</sub></th>"
        "<th colspan='2' class='g2'>MTTF, с</th>"
        "<th colspan='2' class='g2'>Точность</th>"
        "<th colspan='2' class='g2'>Энергия, усл. ед.</th>"
        "<th colspan='2' class='g2'>CMVI, усл. ед.</th>"
        "<th rowspan='2'>ΔCMVI, %</th>"
        "</tr>"
        "<tr>"
        "<th>Q-TCID</th><th>TA-QTCID</th>"
        "<th>Q-TCID</th><th>TA-QTCID</th>"
        "<th>Q-TCID</th><th>TA-QTCID</th>"
        "<th>Q-TCID</th><th>TA-QTCID</th>"
        "</tr>"
    )
    prev_pa = None
    for r in rep_rows:
        pa = float(r["pa"])
        tids = int(r["tids"])
        pa_cell = f'<td class="pa">{pa}</td>' if pa != prev_pa else "<td></td>"
        prev_pa = pa
        ta_cmvi = float(r["ta_cmvi_mean"])
        q_cmvi = float(r["q_cmvi_mean"])
        delta = float(r["cmvi_improvement_pct"])
        cmvi_cls = ' class="better"' if ta_cmvi < q_cmvi else ""
        delta_cls = ' class="better"' if delta > 0 else ""
        out_lines.append(
            f"<tr>{pa_cell}<td>{tids}</td>"
            f"<td>{float(r['q_mttf_mean']):.0f}</td><td>{float(r['ta_mttf_mean']):.0f}</td>"
            f"<td>{float(r['q_accuracy_mean']):.5f}</td><td>{float(r['ta_accuracy_mean']):.5f}</td>"
            f"<td>{float(r['q_energy_spent_mean']):.1f}</td><td>{float(r['ta_energy_spent_mean']):.1f}</td>"
            f"<td>{q_cmvi:.2f}</td><td{cmvi_cls}>{ta_cmvi:.2f}</td>"
            f"<td{delta_cls}>{delta:.1f}</td></tr>"
        )
    out_lines.append("</table>")

    # ── Таблица 2: абсолютные значения CMVI по Pa × TIDS ───────────────────
    out_lines.append("<h3>Таблица 2. Абсолютные значения CMVI для Q-TCID и TA-QTCID (по P<sub>a</sub> и T<sub>IDS</sub>)</h3>")
    out_lines.append("<table>")
    header_cells = "".join(f"<th>T<sub>IDS</sub>={t}</th>" for t in TIDS_VALUES)
    out_lines.append(
        "<tr>"
        "<th rowspan='2'>P<sub>a</sub></th>"
        "<th rowspan='2'>Метод</th>"
        f"<th colspan='{len(TIDS_VALUES)}' class='g2'>CMVI, усл. ед.</th>"
        "<th rowspan='2'>Среднее</th>"
        "</tr>"
        f"<tr>{header_cells}</tr>"
    )
    for pa in PA_VALUES:
        subset = [r for r in detail_rows if float(r["pa"]) == pa]
        by_tids = {int(r["tids"]): r for r in subset}

        q_vals = [float(by_tids[t]["q_cmvi_mean"]) for t in TIDS_VALUES]
        ta_vals = [float(by_tids[t]["ta_cmvi_mean"]) for t in TIDS_VALUES]
        q_avg = mean(q_vals)
        ta_avg = mean(ta_vals)

        q_cells = "".join(f"<td>{v:.2f}</td>" for v in q_vals)
        ta_cells = "".join(
            ("<td class='better'>{:.2f}</td>" if ta_vals[i] < q_vals[i] else "<td>{:.2f}</td>").format(ta_vals[i])
            for i in range(len(TIDS_VALUES))
        )
        avg_cls = ' class="better"' if ta_avg < q_avg else ""

        out_lines.append(
            f"<tr><td class='pa' rowspan='2'>{pa}</td><td>Q-TCID</td>{q_cells}<td>{q_avg:.2f}</td></tr>"
        )
        out_lines.append(
            f"<tr><td>TA-QTCID</td>{ta_cells}<td{avg_cls}><b>{ta_avg:.2f}</b></td></tr>"
        )
    out_lines.append("</table>")

    out_lines.append("</body>\n</html>")

    out = OUT_DIR / "tables_for_word.html"
    out.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"  {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    rep_rows = []
    bvs_rows = []

    for pa in PA_VALUES:
        for tids in TIDS_VALUES:
            print(f"Running: Pa={pa:.2f}, TIDS={tids}")

            q_cfg = build_qtcid_cfg(pa, tids)
            ta_cfg = build_taqtcid_cfg(pa, tids)
            bvs_cfg = build_bvs_cfg(pa, tids)

            q = run_qtcid_family_manual(q_cfg, QTCIDSimulator)
            ta = run_qtcid_family_manual(ta_cfg, TAQTCIDSimulator)
            bvs = run_wang_bvs_monte_carlo(bvs_cfg)

            cmvi_improvement_pct = safe_pct_improvement(q["cmvi_mean"], ta["cmvi_mean"])

            row = {
                "pa": pa,
                "tids": tids,

                "q_mttf_mean": round(q["mttf_mean"], 4),
                "q_mttf_std": round(q["mttf_std"], 4),
                "q_accuracy_mean": round(q["accuracy_mean"], 6),
                "q_energy_spent_mean": round(q["energy_spent_mean"], 4),
                "q_cmvi_mean": round(q["cmvi_mean"], 4),
                "q_cmvi_std": round(q["cmvi_std"], 4),
                "q_false_good_evictions_mean": round(q["false_good_evictions_mean"], 4),
                "q_bad_nodes_retained_mean": round(q["bad_nodes_retained_mean"], 4),
                "q_audit_mismatch_events_mean": round(q["audit_mismatch_events_mean"], 4),

                "ta_mttf_mean": round(ta["mttf_mean"], 4),
                "ta_mttf_std": round(ta["mttf_std"], 4),
                "ta_accuracy_mean": round(ta["accuracy_mean"], 6),
                "ta_energy_spent_mean": round(ta["energy_spent_mean"], 4),
                "ta_cmvi_mean": round(ta["cmvi_mean"], 4),
                "ta_cmvi_std": round(ta["cmvi_std"], 4),
                "ta_false_good_evictions_mean": round(ta["false_good_evictions_mean"], 4),
                "ta_bad_nodes_retained_mean": round(ta["bad_nodes_retained_mean"], 4),
                "ta_audit_mismatch_events_mean": round(ta["audit_mismatch_events_mean"], 4),

                "cmvi_improvement_pct": round(cmvi_improvement_pct, 4),
            }
            detail_rows.append(row)

            bvs_rows.append({
                "pa": pa,
                "tids": tids,
                "bvs_mttf_mean": round(bvs.mttf_mean, 4),
                "bvs_accuracy_mean": round(bvs.accuracy_mean, 6),
                "bvs_energy_spent_mean": round(bvs.energy_spent_mean, 4),
            })

            if tids in REPRESENTATIVE_TIDS:
                rep_rows.append({
                    "pa": pa,
                    "tids": tids,
                    "q_mttf_mean": round(q["mttf_mean"], 4),
                    "q_accuracy_mean": round(q["accuracy_mean"], 6),
                    "q_energy_spent_mean": round(q["energy_spent_mean"], 4),
                    "q_cmvi_mean": round(q["cmvi_mean"], 4),

                    "ta_mttf_mean": round(ta["mttf_mean"], 4),
                    "ta_accuracy_mean": round(ta["accuracy_mean"], 6),
                    "ta_energy_spent_mean": round(ta["energy_spent_mean"], 4),
                    "ta_cmvi_mean": round(ta["cmvi_mean"], 4),

                    "cmvi_improvement_pct": round(cmvi_improvement_pct, 4),
                })

    write_csv(OUT_DIR / "table_qtcid_vs_taqtcid_detailed.csv", detail_rows)
    write_csv(OUT_DIR / "table_qtcid_vs_taqtcid_representative.csv", rep_rows)
    write_csv(OUT_DIR / "table_bvs_baseline.csv", bvs_rows)

    plot_mttf_by_pa(detail_rows)
    plot_cmvi_by_pa(detail_rows)
    plot_cmvi_improvement(detail_rows)
    plot_representative_bars(rep_rows)
    format_word_tables(detail_rows, rep_rows)

    print("\nSaved:")
    print(OUT_DIR / "table_qtcid_vs_taqtcid_detailed.csv")
    print(OUT_DIR / "table_qtcid_vs_taqtcid_representative.csv")
    print(OUT_DIR / "table_bvs_baseline.csv")
    print(OUT_DIR / "fig_mttf_qtcid_vs_taqtcid.png")
    print(OUT_DIR / "fig_cmvi_qtcid_vs_taqtcid.png")
    print(OUT_DIR / "fig_cmvi_improvement_pct.png")
    print(OUT_DIR / "fig_avg_cmvi_improvement_representative.png")


if __name__ == "__main__":
    main()