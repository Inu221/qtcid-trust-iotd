"""
Расширенный эксперимент для статьи: приоритизация системной проверки узлов.

УЛУЧШЕНИЯ ДЛЯ СТАТЬИ:
1. Расширенная сетка параметров: 8 точек по budget, 5 точек по observer_error
2. 30 прогонов для статистической устойчивости
3. Ablation study: current_only, history_no_stability, history_based
4. Heatmap улучшения по budget × observer_error
5. Grouped bar chart для главного сценария
6. Полная русификация всех графиков
7. Confidence intervals на всех графиках
8. Профессиональная визуализация с seaborn
"""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import sys
import time

# Allow running the script directly from `experiments/` without installing the repo
# as a package first.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from game_ext.qtcid_repro.audit_prioritization_core_v2 import (
    AuditPrioritizationConfig,
    run_prioritization_monte_carlo,
)
from game_ext.qtcid_repro.experiments.visualization_article_ru import (
    plot_line_with_ci,
    plot_improvement_heatmap,
    plot_grouped_bar_chart,
    plot_ablation_study,
    plot_main_scenario_panel,
)


# -----------------------------------------------------------------------------
# Режимы работы
# -----------------------------------------------------------------------------

# FAST_MODE: быстрое тестирование (меньше точек, меньше прогонов)
# ARTICLE_MODE: полный эксперимент для статьи

MODE = os.environ.get("AUDIT_PRIORITIZATION_MODE", "FAST").strip().upper()

if MODE == "FAST":
    AUDIT_BUDGET_VALUES = [3, 5, 8]
    OBSERVER_ERROR_VALUES = [0.10, 0.15, 0.20]
    SCENARIOS_TO_RUN = ["medium_noise_mixed"]
    RUNS = 15
    MAX_TIME = 5000
    INCLUDE_ABLATION = True
elif MODE == "ARTICLE":
    AUDIT_BUDGET_VALUES = [2, 3, 4, 5, 6, 7, 8, 10]
    OBSERVER_ERROR_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]
    SCENARIOS_TO_RUN = ["medium_noise_mixed", "low_noise_persistent", "high_noise_intermittent"]
    RUNS = 30
    MAX_TIME = 10000
    INCLUDE_ABLATION = True
else:
    raise ValueError(f"Unknown MODE: {MODE}")

# Сценарии экспериментов
SCENARIOS = {
    "low_noise_persistent": {
        "benign_noise_level": 0.1,
        "intermittent_activity_prob": 0.3,
        "persistent_activity_prob": 0.95,
        "prob_intermittent_on_capture": 0.2,
    },
    "medium_noise_mixed": {
        "benign_noise_level": 0.25,
        "intermittent_activity_prob": 0.5,
        "persistent_activity_prob": 0.9,
        "prob_intermittent_on_capture": 0.5,
    },
    "high_noise_intermittent": {
        "benign_noise_level": 0.4,
        "intermittent_activity_prob": 0.6,
        "persistent_activity_prob": 0.85,
        "prob_intermittent_on_capture": 0.8,
    },
}

# Методы приоритизации
PRIORITIZATION_MODES = ["random", "current_only", "history_based"]

# Для ablation study добавляем history_no_stability
if INCLUDE_ABLATION:
    ABLATION_MODES = ["current_only", "history_no_stability", "history_based"]
else:
    ABLATION_MODES = []

# Параметры симуляции
SEED = 42
N_NODES = 128
N_NEIGHBORS = 32
M_VOTERS = 5
TIDS = 200
PA = 0.5

# Директории для результатов
OUT_DIR = REPO_ROOT / "results" / "audit_prioritization_article"
FIGURES_DIR = OUT_DIR / "figures_article_ru"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Функции построения конфигураций
# -----------------------------------------------------------------------------

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
        # Priority coefficients (можно слегка усилить stability penalty)
        priority_current_disagreement=0.30,
        priority_current_anomaly=0.20,
        priority_ema_mismatch=0.25,
        priority_persistence=0.15,
        priority_stability_penalty=0.40,  # увеличено с 0.35 для снижения false_attention
        current_only_disagreement=0.60,
        current_only_anomaly=0.40,
        # Local voting
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
# Основной эксперимент
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("РАСШИРЕННЫЙ ЭКСПЕРИМЕНТ ДЛЯ СТАТЬИ: ПРИОРИТИЗАЦИЯ СИСТЕМНОЙ ПРОВЕРКИ")
    print("=" * 80)
    print(f"Режим: {MODE}")
    print(f"Прогонов на конфигурацию: {RUNS}")
    print(f"Бюджеты аудита: {AUDIT_BUDGET_VALUES}")
    print(f"Уровни ошибки наблюдателя: {OBSERVER_ERROR_VALUES}")
    print(f"Сценарии: {SCENARIOS_TO_RUN}")
    print(f"Методы: {PRIORITIZATION_MODES}")
    print(f"Ablation study: {INCLUDE_ABLATION}")
    print("=" * 80)

    all_results = []
    start_time = time.time()

    total_configs = (
        len(AUDIT_BUDGET_VALUES)
        * len(SCENARIOS_TO_RUN)
        * len(PRIORITIZATION_MODES)
        * len(OBSERVER_ERROR_VALUES)
    )

    if INCLUDE_ABLATION:
        # Для ablation: запустим только для одного сценария и одного observer_error
        ablation_scenario = "medium_noise_mixed"
        ablation_observer_error = 0.15
        total_configs += len(AUDIT_BUDGET_VALUES) * len(ABLATION_MODES)

    config_counter = 0

    # Основной эксперимент
    for scenario_name in SCENARIOS_TO_RUN:
        for observer_error in OBSERVER_ERROR_VALUES:
            for audit_budget in AUDIT_BUDGET_VALUES:
                for mode in PRIORITIZATION_MODES:
                    config_counter += 1
                    print(
                        f"\n[{config_counter}/{total_configs}] Запуск: "
                        f"scenario={scenario_name}, mode={mode}, budget={audit_budget}, "
                        f"observer_error={observer_error}"
                    )

                    cfg = build_config(mode, audit_budget, scenario_name, observer_error)
                    summary = run_prioritization_monte_carlo(cfg)

                    row = {
                        "scenario": scenario_name,
                        "mode": mode,
                        "audit_budget": audit_budget,
                        "observer_error": observer_error,
                        **{
                            k: round(v, 6) if isinstance(v, float) else v
                            for k, v in summary.items()
                        },
                    }

                    all_results.append(row)
                    print(
                        f"  precision={summary['audit_precision_mean']:.4f}, "
                        f"recall={summary['recall_hit_rate_mean']:.4f}, "
                        f"false_attention={summary['false_attention_rate_mean']:.4f}"
                    )

    # Ablation study
    if INCLUDE_ABLATION:
        print("\n" + "=" * 80)
        print("ABLATION STUDY: влияние компонентов истории")
        print("=" * 80)

        for audit_budget in AUDIT_BUDGET_VALUES:
            for mode in ABLATION_MODES:
                config_counter += 1
                print(
                    f"\n[{config_counter}/{total_configs}] Ablation: "
                    f"mode={mode}, budget={audit_budget}"
                )

                cfg = build_config(
                    mode, audit_budget, ablation_scenario, ablation_observer_error
                )
                summary = run_prioritization_monte_carlo(cfg)

                row = {
                    "scenario": ablation_scenario,
                    "mode": mode,
                    "audit_budget": audit_budget,
                    "observer_error": ablation_observer_error,
                    **{
                        k: round(v, 6) if isinstance(v, float) else v
                        for k, v in summary.items()
                    },
                }

                # Добавляем в результаты, чтобы не было дублирования
                if mode not in PRIORITIZATION_MODES:
                    all_results.append(row)

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Эксперименты завершены за {elapsed / 60:.1f} минут")
    print(f"Всего конфигураций: {total_configs}")
    print(f"Всего прогонов: {total_configs * RUNS}")
    print(f"{'='*80}")

    # Сохранить результаты
    write_csv(OUT_DIR / "detailed_results_article.csv", all_results)
    write_json(OUT_DIR / "detailed_results_article.json", {"results": all_results})

    print("\n" + "=" * 80)
    print("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ ДЛЯ СТАТЬИ")
    print("=" * 80)

    # Главный сценарий
    main_scenario = "medium_noise_mixed"
    main_observer_error = 0.15

    # 1. Главная панель для статьи (6 метрик)
    print("Создание главной панели...")
    plot_main_scenario_panel(
        all_results,
        scenario=main_scenario,
        observer_error=main_observer_error,
        output_path=FIGURES_DIR / "main_scenario_panel.png",
    )

    # 2. Heatmap улучшения по precision
    print("Создание heatmap улучшения по precision...")
    plot_improvement_heatmap(
        all_results,
        metric_key="audit_precision_mean",
        title="Улучшение точности отбора: history-based vs current-only (%)",
        output_path=FIGURES_DIR / "heatmap_precision_improvement.png",
        filter_dict={"scenario": main_scenario},
    )

    # 3. Heatmap улучшения по cumulative_residual_risk (меньше - лучше)
    print("Создание heatmap улучшения по cumulative risk...")
    # Для risk: улучшение = (current - history) / current * 100 (чем меньше risk, тем лучше)
    # Модифицируем для отображения
    plot_improvement_heatmap(
        all_results,
        metric_key="cumulative_residual_risk_mean",
        title="Снижение накопленного риска: history-based vs current-only (%)",
        output_path=FIGURES_DIR / "heatmap_risk_improvement.png",
        filter_dict={"scenario": main_scenario},
    )

    # 4. Grouped bar chart для 3 бюджетов
    print("Создание grouped bar chart...")
    selected_budgets = [3, 5, 8]
    if MODE == "ARTICLE":
        selected_budgets = [3, 5, 8]

    plot_grouped_bar_chart(
        all_results,
        metrics=[
            ("audit_precision_mean", "Точность отбора"),
            ("recall_hit_rate_mean", "Полнота выявления"),
            ("false_attention_rate_mean", "Доля ложного внимания"),
            ("cumulative_residual_risk_mean", "Накопленный риск"),
        ],
        budgets=selected_budgets,
        filter_dict={"scenario": main_scenario, "observer_error": main_observer_error},
        title=f"Сравнение методов: {main_scenario} (observer_error={main_observer_error:.2f})",
        output_path=FIGURES_DIR / "grouped_bar_chart_main_scenario.png",
    )

    # 5. Ablation plot
    if INCLUDE_ABLATION:
        print("Создание ablation plot...")
        plot_ablation_study(
            all_results,
            metric_key="audit_precision_mean",
            filter_dict={
                "scenario": ablation_scenario,
                "observer_error": ablation_observer_error,
            },
            title="Ablation Study: вклад компонентов истории в точность отбора",
            xlabel="Бюджет системной проверки (узлов за цикл)",
            ylabel="Точность отбора узлов на проверку",
            output_path=FIGURES_DIR / "ablation_plot_precision.png",
        )

        plot_ablation_study(
            all_results,
            metric_key="false_attention_rate_mean",
            filter_dict={
                "scenario": ablation_scenario,
                "observer_error": ablation_observer_error,
            },
            title="Ablation Study: влияние стабильности на ложное внимание",
            xlabel="Бюджет системной проверки (узлов за цикл)",
            ylabel="Доля ложного внимания к шумным узлам",
            output_path=FIGURES_DIR / "ablation_plot_false_attention.png",
        )

    # 6. Line plots с CI для разных сценариев
    if len(SCENARIOS_TO_RUN) > 1:
        print("Создание line plots для всех сценариев...")
        for scenario in SCENARIOS_TO_RUN:
            plot_line_with_ci(
                all_results,
                x_key="audit_budget",
                value_key="audit_precision_mean",
                group_key="mode",
                filter_dict={"scenario": scenario, "observer_error": main_observer_error},
                title=f"Точность отбора vs бюджет: {scenario}",
                xlabel="Бюджет системной проверки (узлов за цикл)",
                ylabel="Точность отбора узлов на проверку",
                output_path=FIGURES_DIR / f"line_precision_{scenario}.png",
            )

    # 7. Line plot: влияние observer_error на precision при фиксированном budget
    if len(OBSERVER_ERROR_VALUES) > 2:
        print("Создание line plot: влияние observer_error...")
        # Транспонируем: теперь x = observer_error, group = mode, fixed_budget
        fixed_budget = 5
        plot_line_with_ci(
            all_results,
            x_key="observer_error",
            value_key="audit_precision_mean",
            group_key="mode",
            filter_dict={"scenario": main_scenario, "audit_budget": fixed_budget},
            title=f"Влияние уровня ошибки наблюдателя (budget={fixed_budget})",
            xlabel="Уровень ошибки наблюдателя",
            ylabel="Точность отбора узлов на проверку",
            output_path=FIGURES_DIR / f"line_observer_error_impact.png",
        )

    print("\n" + "=" * 80)
    print("ЗАВЕРШЕНО")
    print("=" * 80)
    print(f"Результаты сохранены в: {OUT_DIR}")
    print(f"Графики для статьи сохранены в: {FIGURES_DIR}")
    print(f"Всего конфигураций: {total_configs}")
    print(f"Всего прогонов: {total_configs * RUNS}")
    print(f"Время выполнения: {elapsed / 60:.1f} минут")
    print("=" * 80)


# -----------------------------------------------------------------------------
# Создание итоговых таблиц
# -----------------------------------------------------------------------------

def create_summary_tables(results_path: Path, output_dir: Path):
    """Создать итоговые таблицы для статьи на русском языке"""
    import json
    import pandas as pd

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    df = pd.DataFrame(results)

    # Таблица 1: Главный сценарий, observer_error=0.15, разные бюджеты
    main_scenario = "medium_noise_mixed"
    main_observer_error = 0.15

    subset = df[
        (df["scenario"] == main_scenario) & (df["observer_error"] == main_observer_error)
    ]

    table1_data = []
    for budget in sorted(subset["audit_budget"].unique()):
        row = {"Бюджет": budget}
        for mode in ["random", "current_only", "history_based"]:
            mode_data = subset[(subset["audit_budget"] == budget) & (subset["mode"] == mode)]
            if len(mode_data) > 0:
                precision = mode_data["audit_precision_mean"].values[0]
                recall = mode_data["recall_hit_rate_mean"].values[0]
                false_attn = mode_data["false_attention_rate_mean"].values[0]

                row[f"{mode}_precision"] = f"{precision:.3f}"
                row[f"{mode}_recall"] = f"{recall:.3f}"
                row[f"{mode}_false_attn"] = f"{false_attn:.3f}"

        table1_data.append(row)

    table1 = pd.DataFrame(table1_data)
    table1.to_csv(output_dir / "table1_main_scenario_ru.csv", index=False, encoding="utf-8")

    # Таблица 2: Улучшение history_based относительно current_only
    table2_data = []
    for budget in sorted(subset["audit_budget"].unique()):
        current_data = subset[
            (subset["audit_budget"] == budget) & (subset["mode"] == "current_only")
        ]
        history_data = subset[
            (subset["audit_budget"] == budget) & (subset["mode"] == "history_based")
        ]

        if len(current_data) > 0 and len(history_data) > 0:
            curr_prec = current_data["audit_precision_mean"].values[0]
            hist_prec = history_data["audit_precision_mean"].values[0]

            improvement = (hist_prec - curr_prec) / curr_prec * 100 if curr_prec > 0 else 0

            table2_data.append({
                "Бюджет": budget,
                "Current-only precision": f"{curr_prec:.3f}",
                "History-based precision": f"{hist_prec:.3f}",
                "Улучшение (%)": f"{improvement:+.1f}",
            })

    table2 = pd.DataFrame(table2_data)
    table2.to_csv(output_dir / "table2_improvement_ru.csv", index=False, encoding="utf-8")

    print(f"\nИтоговые таблицы сохранены в {output_dir}")
    print(f"  - table1_main_scenario_ru.csv")
    print(f"  - table2_improvement_ru.csv")


if __name__ == "__main__":
    main()

    # Создать итоговые таблицы
    results_file = OUT_DIR / "detailed_results_article.json"
    if results_file.exists():
        print("\n" + "=" * 80)
        print("СОЗДАНИЕ ИТОГОВЫХ ТАБЛИЦ")
        print("=" * 80)
        create_summary_tables(results_file, OUT_DIR)
