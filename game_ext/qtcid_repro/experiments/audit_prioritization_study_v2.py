"""
Переработанный эксперимент приоритизации с улучшенными сценариями.

КРИТИЧЕСКИЕ ИЗМЕНЕНИЯ:
1. Использует audit_prioritization_core_v2 БЕЗ утечки ground truth
2. Сценарии: Low/Medium/High noise + Intermittent/Persistent attacks
3. Новые метрики: intermittent_detection_rate, false_attention_rate
4. Ablation study для history components
5. Анализ чувствительности к параметрам
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from game_ext.qtcid_repro.audit_prioritization_core_v2 import (
    AuditPrioritizationConfig,
    AuditPrioritizationSimulator,
    run_prioritization_monte_carlo,
)


# Режимы работы
FAST_MODE = True   # Быстрый прогон для тестирования
FULL_MODE = False  # Полный эксперимент

# Сценарии экспериментов
SCENARIOS = {
    "low_noise_persistent": {
        "benign_noise_level": 0.1,
        "intermittent_activity_prob": 0.3,
        "persistent_activity_prob": 0.95,
        "prob_intermittent_on_capture": 0.2,  # 20% intermittent, 80% persistent
    },
    "medium_noise_mixed": {
        "benign_noise_level": 0.25,
        "intermittent_activity_prob": 0.5,
        "persistent_activity_prob": 0.9,
        "prob_intermittent_on_capture": 0.5,  # 50/50
    },
    "high_noise_intermittent": {
        "benign_noise_level": 0.4,
        "intermittent_activity_prob": 0.6,
        "persistent_activity_prob": 0.85,
        "prob_intermittent_on_capture": 0.8,  # 80% intermittent
    },
}

if FAST_MODE:
    AUDIT_BUDGET_VALUES = [3, 5, 8]
    OBSERVER_ERROR_VALUES = [0.15]
    SCENARIOS_TO_RUN = ["medium_noise_mixed"]
    RUNS = 10
    MAX_TIME = 5000
else:
    AUDIT_BUDGET_VALUES = [3, 5, 8, 12, 16]
    OBSERVER_ERROR_VALUES = [0.10, 0.15, 0.20]
    SCENARIOS_TO_RUN = list(SCENARIOS.keys())
    RUNS = 40
    MAX_TIME = 10000

PRIORITIZATION_MODES = ["random", "current_only", "history_based"]

SEED = 42
N_NODES = 128
N_NEIGHBORS = 32
M_VOTERS = 5
TIDS = 200
PA = 0.5

OUT_DIR = Path("results/audit_prioritization_study_v2")
FIGURES_DIR = OUT_DIR / "figures"


def build_config(
    mode: str,
    audit_budget: int,
    scenario_name: str,
    observer_error: float = 0.15,
    runs: int = RUNS,
) -> AuditPrioritizationConfig:
    """Создать конфигурацию для эксперимента"""
    scenario_params = SCENARIOS[scenario_name]

    return AuditPrioritizationConfig(
        n_nodes=N_NODES,
        n_neighbors=N_NEIGHBORS,
        m_voters=M_VOTERS,
        tids=TIDS,
        lambda_capture=1 / 3600,
        hpfn=0.05,
        hpfp=0.05,
        pa=PA,
        pc=0.5,
        beta=1.0,
        runs=runs,
        max_time=MAX_TIME,
        seed=SEED,
        prioritization_mode=mode,
        audit_budget_fixed=audit_budget,
        audit_budget_fraction=0.0,
        # Priority coefficients
        priority_current_disagreement=0.30,
        priority_current_anomaly=0.20,
        priority_ema_mismatch=0.25,
        priority_persistence=0.15,
        priority_stability_penalty=0.35,
        current_only_disagreement=0.60,
        current_only_anomaly=0.40,
        # Locал voting
        n_local_observers=5,
        observer_error_rate=observer_error,
        # Scenario params
        **scenario_params,
        prob_benign_noisy=0.3,
        history_window=10,
        ema_decay=0.7,
        alarm_threshold=0.5,
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

SCENARIO_LABELS = {
    "low_noise_persistent": "Low Noise, Persistent Attacks",
    "medium_noise_mixed": "Medium Noise, Mixed Attacks",
    "high_noise_intermittent": "High Noise, Intermittent Attacks",
}


def plot_audit_precision_vs_budget(results: list[dict], scenario: str) -> None:
    """График: audit precision vs budget"""
    subset = [r for r in results if r["scenario"] == scenario]

    fig, ax = plt.subplots(figsize=(9, 6))

    for mode in PRIORITIZATION_MODES:
        mode_data = [r for r in subset if r["mode"] == mode]
        if not mode_data:
            continue

        mode_data.sort(key=lambda x: x["audit_budget"])
        x = [r["audit_budget"] for r in mode_data]
        y = [r["audit_precision_mean"] for r in mode_data]

        ax.plot(x, y, marker="o", linewidth=2.5, color=COLORS[mode], label=LABELS[mode])

    ax.set_xlabel("Audit Budget (nodes per cycle)", fontsize=12)
    ax.set_ylabel("Audit Precision", fontsize=12)
    ax.set_title(f"Audit Precision vs Budget\n{SCENARIO_LABELS[scenario]}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"audit_precision_vs_budget_{scenario}.png", dpi=260)
    plt.close()


def plot_recall_vs_budget(results: list[dict], scenario: str) -> None:
    """График: recall (hit rate) vs budget"""
    subset = [r for r in results if r["scenario"] == scenario]

    fig, ax = plt.subplots(figsize=(9, 6))

    for mode in PRIORITIZATION_MODES:
        mode_data = [r for r in subset if r["mode"] == mode]
        if not mode_data:
            continue

        mode_data.sort(key=lambda x: x["audit_budget"])
        x = [r["audit_budget"] for r in mode_data]
        y = [r["recall_hit_rate_mean"] for r in mode_data]

        ax.plot(x, y, marker="o", linewidth=2.5, color=COLORS[mode], label=LABELS[mode])

    ax.set_xlabel("Audit Budget (nodes per cycle)", fontsize=12)
    ax.set_ylabel("Recall (Hit Rate)", fontsize=12)
    ax.set_title(f"Recall vs Budget\n{SCENARIO_LABELS[scenario]}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"recall_vs_budget_{scenario}.png", dpi=260)
    plt.close()


def plot_intermittent_detection_vs_budget(results: list[dict], scenario: str) -> None:
    """График: intermittent detection rate vs budget"""
    subset = [r for r in results if r["scenario"] == scenario]

    fig, ax = plt.subplots(figsize=(9, 6))

    for mode in PRIORITIZATION_MODES:
        mode_data = [r for r in subset if r["mode"] == mode]
        if not mode_data:
            continue

        mode_data.sort(key=lambda x: x["audit_budget"])
        x = [r["audit_budget"] for r in mode_data]
        y = [r["intermittent_detection_rate_mean"] for r in mode_data]

        ax.plot(x, y, marker="o", linewidth=2.5, color=COLORS[mode], label=LABELS[mode])

    ax.set_xlabel("Audit Budget (nodes per cycle)", fontsize=12)
    ax.set_ylabel("Intermittent Detection Rate", fontsize=12)
    ax.set_title(f"Intermittent Malicious Detection Rate\n{SCENARIO_LABELS[scenario]}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"intermittent_detection_{scenario}.png", dpi=260)
    plt.close()


def plot_false_attention_vs_budget(results: list[dict], scenario: str) -> None:
    """График: false attention rate vs budget"""
    subset = [r for r in results if r["scenario"] == scenario]

    fig, ax = plt.subplots(figsize=(9, 6))

    for mode in PRIORITIZATION_MODES:
        mode_data = [r for r in subset if r["mode"] == mode]
        if not mode_data:
            continue

        mode_data.sort(key=lambda x: x["audit_budget"])
        x = [r["audit_budget"] for r in mode_data]
        y = [r["false_attention_rate_mean"] for r in mode_data]

        ax.plot(x, y, marker="o", linewidth=2.5, color=COLORS[mode], label=LABELS[mode])

    ax.set_xlabel("Audit Budget (nodes per cycle)", fontsize=12)
    ax.set_ylabel("False Attention Rate (to benign_noisy)", fontsize=12)
    ax.set_title(f"False Attention Rate\n{SCENARIO_LABELS[scenario]}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"false_attention_{scenario}.png", dpi=260)
    plt.close()


def plot_cumulative_risk_vs_budget(results: list[dict], scenario: str) -> None:
    """График: cumulative residual risk vs budget"""
    subset = [r for r in results if r["scenario"] == scenario]

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
    ax.set_title(f"Cumulative Residual Risk vs Budget\n{SCENARIO_LABELS[scenario]}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"cumulative_risk_{scenario}.png", dpi=260)
    plt.close()


def plot_mean_cycles_intermittent_vs_budget(results: list[dict], scenario: str) -> None:
    """График: mean cycles to verify intermittent vs budget"""
    subset = [r for r in results if r["scenario"] == scenario]

    fig, ax = plt.subplots(figsize=(9, 6))

    for mode in PRIORITIZATION_MODES:
        mode_data = [r for r in subset if r["mode"] == mode]
        if not mode_data:
            continue

        mode_data.sort(key=lambda x: x["audit_budget"])
        x = [r["audit_budget"] for r in mode_data]
        y = [r["mean_cycles_to_verify_intermittent"] for r in mode_data]

        ax.plot(x, y, marker="o", linewidth=2.5, color=COLORS[mode], label=LABELS[mode])

    ax.set_xlabel("Audit Budget (nodes per cycle)", fontsize=12)
    ax.set_ylabel("Mean Cycles to Verify (Intermittent)", fontsize=12)
    ax.set_title(f"Delay to Detect Intermittent Attacks\n{SCENARIO_LABELS[scenario]}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"mean_cycles_intermittent_{scenario}.png", dpi=260)
    plt.close()


def plot_scenarios_comparison(results: list[dict], fixed_budget: int) -> None:
    """Сравнительный график для всех сценариев при фиксированном бюджете"""
    if len(SCENARIOS_TO_RUN) < 2:
        return

    subset = [r for r in results if r["audit_budget"] == fixed_budget]

    scenarios = sorted(set(r["scenario"] for r in subset))
    x_pos = np.arange(len(scenarios))
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Audit Precision
    ax = axes[0, 0]
    for i, mode in enumerate(PRIORITIZATION_MODES):
        values = [next((r["audit_precision_mean"] for r in subset if r["scenario"] == s and r["mode"] == mode), 0) for s in scenarios]
        ax.bar(x_pos + i * width, values, width, label=LABELS[mode], color=COLORS[mode])
    ax.set_xlabel("Scenario", fontsize=11)
    ax.set_ylabel("Audit Precision", fontsize=11)
    ax.set_title(f"Audit Precision (Budget={fixed_budget})", fontsize=12)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Intermittent Detection Rate
    ax = axes[0, 1]
    for i, mode in enumerate(PRIORITIZATION_MODES):
        values = [next((r["intermittent_detection_rate_mean"] for r in subset if r["scenario"] == s and r["mode"] == mode), 0) for s in scenarios]
        ax.bar(x_pos + i * width, values, width, label=LABELS[mode], color=COLORS[mode])
    ax.set_xlabel("Scenario", fontsize=11)
    ax.set_ylabel("Intermittent Detection Rate", fontsize=11)
    ax.set_title(f"Intermittent Detection (Budget={fixed_budget})", fontsize=12)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # False Attention Rate
    ax = axes[1, 0]
    for i, mode in enumerate(PRIORITIZATION_MODES):
        values = [next((r["false_attention_rate_mean"] for r in subset if r["scenario"] == s and r["mode"] == mode), 0) for s in scenarios]
        ax.bar(x_pos + i * width, values, width, label=LABELS[mode], color=COLORS[mode])
    ax.set_xlabel("Scenario", fontsize=11)
    ax.set_ylabel("False Attention Rate", fontsize=11)
    ax.set_title(f"False Attention (Budget={fixed_budget})", fontsize=12)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Cumulative Residual Risk
    ax = axes[1, 1]
    for i, mode in enumerate(PRIORITIZATION_MODES):
        values = [next((r["cumulative_residual_risk_mean"] for r in subset if r["scenario"] == s and r["mode"] == mode), 0) for s in scenarios]
        ax.bar(x_pos + i * width, values, width, label=LABELS[mode], color=COLORS[mode])
    ax.set_xlabel("Scenario", fontsize=11)
    ax.set_ylabel("Cumulative Residual Risk", fontsize=11)
    ax.set_title(f"Cumulative Risk (Budget={fixed_budget})", fontsize=12)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"scenarios_comparison_budget{fixed_budget}.png", dpi=260)
    plt.close()


# -----------------------------------------------------------------------------
# Основной эксперимент
# -----------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AUDIT PRIORITIZATION STUDY V2 (REVISED)")
    print("=" * 80)
    print(f"Mode: {'FAST' if FAST_MODE else 'FULL'}")
    print(f"Runs per config: {RUNS}")
    print(f"Audit budgets: {AUDIT_BUDGET_VALUES}")
    print(f"Scenarios: {SCENARIOS_TO_RUN}")
    print(f"Modes: {PRIORITIZATION_MODES}")
    print("=" * 80)

    all_results = []

    total_configs = len(AUDIT_BUDGET_VALUES) * len(SCENARIOS_TO_RUN) * len(PRIORITIZATION_MODES) * len(OBSERVER_ERROR_VALUES)
    config_counter = 0

    for scenario_name in SCENARIOS_TO_RUN:
        for audit_budget in AUDIT_BUDGET_VALUES:
            for observer_error in OBSERVER_ERROR_VALUES:
                for mode in PRIORITIZATION_MODES:
                    config_counter += 1
                    print(f"\n[{config_counter}/{total_configs}] Running: {scenario_name}, mode={mode}, budget={audit_budget}, observer_error={observer_error}")

                    cfg = build_config(mode, audit_budget, scenario_name, observer_error)
                    summary = run_prioritization_monte_carlo(cfg)

                    row = {
                        "scenario": scenario_name,
                        "mode": mode,
                        "audit_budget": audit_budget,
                        "observer_error": observer_error,
                        **{k: round(v, 6) if isinstance(v, float) else v for k, v in summary.items()}
                    }

                    all_results.append(row)
                    print(f"  precision={summary['audit_precision_mean']:.4f}, "
                          f"recall={summary['recall_hit_rate_mean']:.4f}, "
                          f"intermittent_detect={summary['intermittent_detection_rate_mean']:.4f}, "
                          f"false_attention={summary['false_attention_rate_mean']:.4f}")

    # Сохранить результаты
    write_csv(OUT_DIR / "detailed_results_v2.csv", all_results)
    write_json(OUT_DIR / "detailed_results_v2.json", {"results": all_results})

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    typical_budget = AUDIT_BUDGET_VALUES[len(AUDIT_BUDGET_VALUES) // 2] if len(AUDIT_BUDGET_VALUES) > 1 else AUDIT_BUDGET_VALUES[0]

    for scenario in SCENARIOS_TO_RUN:
        print(f"Plotting for scenario: {scenario}")
        plot_audit_precision_vs_budget(all_results, scenario)
        plot_recall_vs_budget(all_results, scenario)
        plot_intermittent_detection_vs_budget(all_results, scenario)
        plot_false_attention_vs_budget(all_results, scenario)
        plot_cumulative_risk_vs_budget(all_results, scenario)
        plot_mean_cycles_intermittent_vs_budget(all_results, scenario)

    # Сравнение сценариев
    if len(SCENARIOS_TO_RUN) > 1:
        print("Plotting scenarios comparison")
        plot_scenarios_comparison(all_results, typical_budget)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED (V2)")
    print("=" * 80)
    print(f"Results saved to: {OUT_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Total configurations: {total_configs}")
    print(f"Total runs: {total_configs * RUNS}")
    print("=" * 80)


if __name__ == "__main__":
    main()
