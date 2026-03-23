"""
Эксперимент: Сравнение методов приоритизации системной проверки узлов
в распределённой сети беспилотных агентов.

Цель: показать преимущество history-based приоритизации над baseline методами
при ограниченном бюджете аудита.

Режимы:
1. random - случайный выбор узлов для аудита
2. current_only - выбор по текущей подозрительности (без истории)
3. history_based - предлагаемый метод с учётом истории согласованности

Метрики:
- hit_rate_at_budget: доля проблемных узлов в аудите
- residual_risk: непроверенные проблемные узлы
- cumulative_residual_risk: накопленный остаточный риск
- mean_cycles_to_verify: среднее время до проверки bad узла
- wasted_audits: проверки корректных узлов
- precision: точность выбора узлов для аудита
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from game_ext.qtcid_repro.audit_prioritization_core import (
    AuditPrioritizationConfig,
    AuditPrioritizationSimulator,
    run_prioritization_monte_carlo,
)


# Режимы работы
FAST_MODE = False  # Быстрый прогон для тестирования
FULL_MODE = True   # Полный эксперимент

if FAST_MODE:
    PA_VALUES = [0.5]
    AUDIT_BUDGET_VALUES = [3, 5, 8]
    TIDS_VALUES = [200]
    RUNS = 10
    MAX_TIME = 5000
elif FULL_MODE:
    PA_VALUES = [0.25, 0.5, 0.75]
    AUDIT_BUDGET_VALUES = [3, 5, 8, 12, 16]
    TIDS_VALUES = [200, 600, 1000]
    RUNS = 60
    MAX_TIME = 12000
else:
    # Средний режим
    PA_VALUES = [0.5]
    AUDIT_BUDGET_VALUES = [5, 10, 15]
    TIDS_VALUES = [200, 600]
    RUNS = 30
    MAX_TIME = 8000

PRIORITIZATION_MODES = ["random", "current_only", "history_based"]

SEED = 42
N_NODES = 128
N_NEIGHBORS = 32
M_VOTERS = 5

OUT_DIR = Path("results/audit_prioritization_study")
FIGURES_DIR = OUT_DIR / "figures"


def build_config(
    mode: str,
    audit_budget: int,
    pa: float,
    tids: int,
    runs: int = RUNS,
) -> AuditPrioritizationConfig:
    """Создать конфигурацию для эксперимента"""
    return AuditPrioritizationConfig(
        n_nodes=N_NODES,
        n_neighbors=N_NEIGHBORS,
        m_voters=M_VOTERS,
        tids=tids,
        lambda_capture=1 / 3600,
        hpfn=0.05,
        hpfp=0.05,
        pa=pa,
        pc=0.5,  # Не используется в новой модели
        beta=1.0,
        runs=runs,
        max_time=MAX_TIME,
        seed=SEED,
        prioritization_mode=mode,
        audit_budget_fixed=audit_budget,
        audit_budget_fraction=0.0,
        # Priority coefficients (можно варьировать)
        priority_disagreement_weight=0.35,
        priority_mismatch_weight=0.30,
        priority_neighbor_weight=0.15,
        priority_anomaly_weight=0.20,
        priority_stability_penalty=0.25,
        history_window=10,
        simulated_disagreement_pa=0.5,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    """Сохранить CSV таблицу"""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    """Сохранить JSON файл"""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Визуализация
# -----------------------------------------------------------------------------

COLORS = {
    "random": "#ff7f0e",
    "current_only": "#2ca02c",
    "history_based": "#d62728",
}

LABELS = {
    "random": "Random",
    "current_only": "Current-only",
    "history_based": "History-based (proposed)",
}


def plot_hit_rate_vs_budget(results: list[dict], fixed_pa: float, fixed_tids: int) -> None:
    """График: hit_rate vs audit_budget"""
    subset = [r for r in results if r["pa"] == fixed_pa and r["tids"] == fixed_tids]

    fig, ax = plt.subplots(figsize=(9, 6))

    for mode in PRIORITIZATION_MODES:
        mode_data = [r for r in subset if r["mode"] == mode]
        if not mode_data:
            continue

        mode_data.sort(key=lambda x: x["audit_budget"])
        x = [r["audit_budget"] for r in mode_data]
        y = [r["hit_rate_mean"] for r in mode_data]

        ax.plot(x, y, marker="o", linewidth=2.5, color=COLORS[mode], label=LABELS[mode])

    ax.set_xlabel("Audit Budget (nodes per cycle)", fontsize=12)
    ax.set_ylabel("Hit Rate (fraction)", fontsize=12)
    ax.set_title(f"Hit Rate vs Audit Budget (Pa={fixed_pa}, TIDS={fixed_tids})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"hit_rate_vs_budget_pa{fixed_pa}_tids{fixed_tids}.png", dpi=260)
    plt.close()


def plot_cumulative_residual_risk_vs_budget(results: list[dict], fixed_pa: float, fixed_tids: int) -> None:
    """График: cumulative_residual_risk vs audit_budget"""
    subset = [r for r in results if r["pa"] == fixed_pa and r["tids"] == fixed_tids]

    fig, ax = plt.subplots(figsize=(9, 6))

    for mode in PRIORITIZATION_MODES:
        mode_data = [r for r in subset if r["mode"] == mode]
        if not mode_data:
            continue

        mode_data.sort(key=lambda x: x["audit_budget"])
        x = [r["audit_budget"] for r in mode_data]
        y = [r["cumulative_residual_risk_mean"] for r in mode_data]

        ax.plot(x, y, marker="o", linewidth=2.5, color=COLORS[mode], label=LABELS[mode])

    ax.set_xlabel("Audit Budget (nodes per cycle)", fontsize=12)
    ax.set_ylabel("Cumulative Residual Risk", fontsize=12)
    ax.set_title(f"Cumulative Residual Risk vs Audit Budget (Pa={fixed_pa}, TIDS={fixed_tids})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"cumulative_risk_vs_budget_pa{fixed_pa}_tids{fixed_tids}.png", dpi=260)
    plt.close()


def plot_mean_cycles_to_verify_vs_pa(results: list[dict], fixed_budget: int, fixed_tids: int) -> None:
    """График: mean_cycles_to_verify vs Pa"""
    subset = [r for r in results if r["audit_budget"] == fixed_budget and r["tids"] == fixed_tids]

    fig, ax = plt.subplots(figsize=(9, 6))

    for mode in PRIORITIZATION_MODES:
        mode_data = [r for r in subset if r["mode"] == mode]
        if not mode_data:
            continue

        mode_data.sort(key=lambda x: x["pa"])
        x = [r["pa"] for r in mode_data]
        y = [r["mean_cycles_to_verify"] for r in mode_data]

        ax.plot(x, y, marker="o", linewidth=2.5, color=COLORS[mode], label=LABELS[mode])

    ax.set_xlabel("Attack Probability (Pa)", fontsize=12)
    ax.set_ylabel("Mean Cycles to Verify Bad Node", fontsize=12)
    ax.set_title(f"Mean Cycles to Verify vs Pa (Budget={fixed_budget}, TIDS={fixed_tids})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"mean_cycles_vs_pa_budget{fixed_budget}_tids{fixed_tids}.png", dpi=260)
    plt.close()


def plot_wasted_audits_vs_budget(results: list[dict], fixed_pa: float, fixed_tids: int) -> None:
    """График: wasted_audits vs audit_budget"""
    subset = [r for r in results if r["pa"] == fixed_pa and r["tids"] == fixed_tids]

    fig, ax = plt.subplots(figsize=(9, 6))

    for mode in PRIORITIZATION_MODES:
        mode_data = [r for r in subset if r["mode"] == mode]
        if not mode_data:
            continue

        mode_data.sort(key=lambda x: x["audit_budget"])
        x = [r["audit_budget"] for r in mode_data]
        y = [r["audits_on_good_mean"] for r in mode_data]

        ax.plot(x, y, marker="o", linewidth=2.5, color=COLORS[mode], label=LABELS[mode])

    ax.set_xlabel("Audit Budget (nodes per cycle)", fontsize=12)
    ax.set_ylabel("Wasted Audits (on good nodes)", fontsize=12)
    ax.set_title(f"Wasted Audits vs Audit Budget (Pa={fixed_pa}, TIDS={fixed_tids})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"wasted_audits_vs_budget_pa{fixed_pa}_tids{fixed_tids}.png", dpi=260)
    plt.close()


def plot_precision_vs_budget(results: list[dict], fixed_pa: float, fixed_tids: int) -> None:
    """График: precision vs audit_budget"""
    subset = [r for r in results if r["pa"] == fixed_pa and r["tids"] == fixed_tids]

    fig, ax = plt.subplots(figsize=(9, 6))

    for mode in PRIORITIZATION_MODES:
        mode_data = [r for r in subset if r["mode"] == mode]
        if not mode_data:
            continue

        mode_data.sort(key=lambda x: x["audit_budget"])
        x = [r["audit_budget"] for r in mode_data]
        y = [r["precision_mean"] for r in mode_data]

        ax.plot(x, y, marker="o", linewidth=2.5, color=COLORS[mode], label=LABELS[mode])

    ax.set_xlabel("Audit Budget (nodes per cycle)", fontsize=12)
    ax.set_ylabel("Precision (audit selection)", fontsize=12)
    ax.set_title(f"Precision of Audit Selection vs Budget (Pa={fixed_pa}, TIDS={fixed_tids})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"precision_vs_budget_pa{fixed_pa}_tids{fixed_tids}.png", dpi=260)
    plt.close()


def plot_residual_risk_dynamics_example(cfg: AuditPrioritizationConfig) -> None:
    """
    График: пример динамики residual_risk по циклам для одного прогона.
    Показывает все три режима на одном графике.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    for mode in PRIORITIZATION_MODES:
        cfg.prioritization_mode = mode
        sim = AuditPrioritizationSimulator(cfg, seed=SEED)
        result = sim.run()

        cycles = list(range(1, len(result.residual_risk_per_cycle) + 1))
        risk = result.residual_risk_per_cycle

        ax.plot(cycles, risk, linewidth=2, color=COLORS[mode], label=LABELS[mode], alpha=0.85)

    ax.set_xlabel("Cycle", fontsize=12)
    ax.set_ylabel("Residual Risk (unverified bad nodes)", fontsize=12)
    ax.set_title(f"Residual Risk Dynamics (Pa={cfg.pa}, Budget={cfg.audit_budget_fixed}, TIDS={cfg.tids})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"residual_risk_dynamics_example_pa{cfg.pa}_budget{cfg.audit_budget_fixed}.png", dpi=260)
    plt.close()


def plot_comparative_heatmap(results: list[dict], mode: str, fixed_tids: int) -> None:
    """
    Heatmap: cumulative_residual_risk для заданного режима
    по осям (Pa, audit_budget).
    """
    subset = [r for r in results if r["mode"] == mode and r["tids"] == fixed_tids]
    if not subset:
        return

    pa_values = sorted(set(r["pa"] for r in subset))
    budget_values = sorted(set(r["audit_budget"] for r in subset))

    matrix = np.zeros((len(pa_values), len(budget_values)))

    for i, pa in enumerate(pa_values):
        for j, budget in enumerate(budget_values):
            row = next((r for r in subset if r["pa"] == pa and r["audit_budget"] == budget), None)
            if row:
                matrix[i, j] = row["cumulative_residual_risk_mean"]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", origin="lower")

    ax.set_xticks(range(len(budget_values)))
    ax.set_xticklabels(budget_values)
    ax.set_yticks(range(len(pa_values)))
    ax.set_yticklabels([f"{pa:.2f}" for pa in pa_values])

    ax.set_xlabel("Audit Budget (nodes per cycle)", fontsize=12)
    ax.set_ylabel("Attack Probability (Pa)", fontsize=12)
    ax.set_title(f"Cumulative Residual Risk Heatmap: {LABELS[mode]} (TIDS={fixed_tids})", fontsize=13)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cumulative Residual Risk", fontsize=11)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"heatmap_{mode}_tids{fixed_tids}.png", dpi=260)
    plt.close()


def plot_improvement_comparison(results: list[dict], fixed_tids: int) -> None:
    """
    Сравнительный график: improvement history_based vs random/current_only
    по cumulative_residual_risk в зависимости от audit_budget.
    """
    subset = [r for r in results if r["tids"] == fixed_tids]

    pa_values = sorted(set(r["pa"] for r in subset))

    fig, axes = plt.subplots(1, len(pa_values), figsize=(16, 5), sharey=True)

    if len(pa_values) == 1:
        axes = [axes]

    for idx, pa in enumerate(pa_values):
        ax = axes[idx]
        pa_data = [r for r in subset if r["pa"] == pa]

        budgets = sorted(set(r["audit_budget"] for r in pa_data))

        # Compute improvement: (baseline - proposed) / baseline * 100
        for baseline_mode in ["random", "current_only"]:
            improvements = []
            for budget in budgets:
                baseline_row = next((r for r in pa_data if r["mode"] == baseline_mode and r["audit_budget"] == budget), None)
                proposed_row = next((r for r in pa_data if r["mode"] == "history_based" and r["audit_budget"] == budget), None)

                if baseline_row and proposed_row:
                    baseline_val = baseline_row["cumulative_residual_risk_mean"]
                    proposed_val = proposed_row["cumulative_residual_risk_mean"]
                    if baseline_val > 0:
                        improvement = (baseline_val - proposed_val) / baseline_val * 100
                    else:
                        improvement = 0
                    improvements.append(improvement)
                else:
                    improvements.append(0)

            label = f"vs {LABELS[baseline_mode]}"
            ax.plot(budgets, improvements, marker="o", linewidth=2.2, label=label)

        ax.set_xlabel("Audit Budget", fontsize=11)
        ax.set_ylabel("Improvement (%)", fontsize=11)
        ax.set_title(f"Pa={pa}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    fig.suptitle(f"Cumulative Residual Risk Improvement: History-based vs Baselines (TIDS={fixed_tids})", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"improvement_comparison_tids{fixed_tids}.png", dpi=260)
    plt.close()


# -----------------------------------------------------------------------------
# Основной эксперимент
# -----------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AUDIT PRIORITIZATION STUDY")
    print("=" * 80)
    print(f"Mode: {'FAST' if FAST_MODE else 'FULL'}")
    print(f"Runs per config: {RUNS}")
    print(f"Pa values: {PA_VALUES}")
    print(f"Audit budgets: {AUDIT_BUDGET_VALUES}")
    print(f"TIDS values: {TIDS_VALUES}")
    print(f"Modes: {PRIORITIZATION_MODES}")
    print("=" * 80)

    all_results = []

    total_configs = len(PA_VALUES) * len(AUDIT_BUDGET_VALUES) * len(TIDS_VALUES) * len(PRIORITIZATION_MODES)
    config_counter = 0

    for pa in PA_VALUES:
        for audit_budget in AUDIT_BUDGET_VALUES:
            for tids in TIDS_VALUES:
                for mode in PRIORITIZATION_MODES:
                    config_counter += 1
                    print(f"\n[{config_counter}/{total_configs}] Running: mode={mode}, budget={audit_budget}, pa={pa}, tids={tids}")

                    cfg = build_config(mode, audit_budget, pa, tids)
                    summary = run_prioritization_monte_carlo(cfg)

                    row = {
                        "mode": mode,
                        "audit_budget": audit_budget,
                        "pa": pa,
                        "tids": tids,
                        **{k: round(v, 6) if isinstance(v, float) else v for k, v in summary.items()}
                    }

                    all_results.append(row)
                    print(f"  hit_rate={summary['hit_rate_mean']:.4f}, "
                          f"cum_risk={summary['cumulative_residual_risk_mean']:.2f}, "
                          f"precision={summary['precision_mean']:.4f}")

    # Сохранить детальные результаты
    write_csv(OUT_DIR / "detailed_results.csv", all_results)
    write_json(OUT_DIR / "detailed_results.json", {"results": all_results})

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Выбрать типичные параметры для графиков
    typical_pa = PA_VALUES[len(PA_VALUES) // 2] if len(PA_VALUES) > 1 else PA_VALUES[0]
    typical_tids = TIDS_VALUES[len(TIDS_VALUES) // 2] if len(TIDS_VALUES) > 1 else TIDS_VALUES[0]
    typical_budget = AUDIT_BUDGET_VALUES[len(AUDIT_BUDGET_VALUES) // 2] if len(AUDIT_BUDGET_VALUES) > 1 else AUDIT_BUDGET_VALUES[0]

    # График 1: Hit rate vs budget
    print("Plotting: hit_rate_vs_budget")
    plot_hit_rate_vs_budget(all_results, typical_pa, typical_tids)

    # График 2: Cumulative residual risk vs budget
    print("Plotting: cumulative_residual_risk_vs_budget")
    plot_cumulative_residual_risk_vs_budget(all_results, typical_pa, typical_tids)

    # График 3: Mean cycles to verify vs Pa
    if len(PA_VALUES) > 1:
        print("Plotting: mean_cycles_to_verify_vs_pa")
        plot_mean_cycles_to_verify_vs_pa(all_results, typical_budget, typical_tids)

    # График 4: Wasted audits vs budget
    print("Plotting: wasted_audits_vs_budget")
    plot_wasted_audits_vs_budget(all_results, typical_pa, typical_tids)

    # График 5: Precision vs budget
    print("Plotting: precision_vs_budget")
    plot_precision_vs_budget(all_results, typical_pa, typical_tids)

    # График 6: Residual risk dynamics (example single run)
    print("Plotting: residual_risk_dynamics_example")
    example_cfg = build_config("history_based", typical_budget, typical_pa, typical_tids, runs=1)
    plot_residual_risk_dynamics_example(example_cfg)

    # График 7: Heatmap для history_based
    if len(PA_VALUES) > 1 and len(AUDIT_BUDGET_VALUES) > 1:
        print("Plotting: heatmap_history_based")
        plot_comparative_heatmap(all_results, "history_based", typical_tids)

    # График 8: Improvement comparison
    if len(AUDIT_BUDGET_VALUES) > 1:
        print("Plotting: improvement_comparison")
        plot_improvement_comparison(all_results, typical_tids)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {OUT_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Total configurations: {total_configs}")
    print(f"Total runs: {total_configs * RUNS}")
    print("=" * 80)


if __name__ == "__main__":
    main()
