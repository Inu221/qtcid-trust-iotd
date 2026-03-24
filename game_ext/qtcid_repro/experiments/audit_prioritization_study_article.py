from __future__ import annotations

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
    build_final_precision_table,
    plot_article_figure_1_heatmap,
    plot_article_figure_2_main_panel,
    plot_article_figure_3_ablation,
)


# -----------------------------------------------------------------------------
# Режимы работы
# -----------------------------------------------------------------------------

MODE = os.environ.get("AUDIT_PRIORITIZATION_MODE", "FAST").strip().upper()
RENDER_ONLY = os.environ.get("AUDIT_PRIORITIZATION_RENDER_ONLY", "0").strip() == "1"

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

PRIORITIZATION_MODES = ["random", "current_only", "history_based"]
ABLATION_MODES = ["current_only", "history_no_stability", "history_based"] if INCLUDE_ABLATION else []

SEED = 42
N_NODES = 128
N_NEIGHBORS = 32
M_VOTERS = 5
TIDS = 200
PA = 0.5

MAIN_SCENARIO = "medium_noise_mixed"
MAIN_OBSERVER_ERROR = 0.15

OUT_DIR = REPO_ROOT / "results" / "audit_prioritization_article"
FINAL_FIGURES_DIR = OUT_DIR / "figures_article_final_ru"
FINAL_TABLES_DIR = OUT_DIR / "tables_article_final_ru"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
FINAL_TABLES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Конфигурации и сохранение данных
# -----------------------------------------------------------------------------

def build_config(
    mode: str,
    audit_budget: int,
    scenario_name: str,
    observer_error: float = 0.15,
    runs: int = RUNS,
) -> AuditPrioritizationConfig:
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
        priority_current_disagreement=0.30,
        priority_current_anomaly=0.20,
        priority_ema_mismatch=0.25,
        priority_persistence=0.15,
        priority_stability_penalty=0.40,
        current_only_disagreement=0.60,
        current_only_anomaly=0.40,
        n_local_observers=5,
        observer_error_rate=observer_error,
        **scenario_params,
        prob_benign_noisy=0.3,
        history_window=10,
        ema_decay=0.7,
        alarm_threshold=0.5,
    )



def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2, ensure_ascii=False)



def load_results(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    return data["results"]


# -----------------------------------------------------------------------------
# Финальные артефакты статьи
# -----------------------------------------------------------------------------

def render_final_article_assets(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("ПОДГОТОВКА ФИНАЛЬНЫХ ИЛЛЮСТРАЦИЙ И ТАБЛИЦ ДЛЯ СТАТЬИ")
    print("=" * 80)

    figure_1 = FINAL_FIGURES_DIR / "figure_1_precision_improvement_heatmap.png"
    figure_2 = FINAL_FIGURES_DIR / "figure_2_main_scenario_metrics.png"
    figure_3 = FINAL_FIGURES_DIR / "figure_3_ablation_history_components.png"
    table_1 = FINAL_TABLES_DIR / "table_1_precision_comparison_main_scenario_ru.csv"

    print("Создание рисунка 1...")
    plot_article_figure_1_heatmap(
        results,
        scenario=MAIN_SCENARIO,
        output_path=figure_1,
    )

    print("Создание рисунка 2...")
    plot_article_figure_2_main_panel(
        results,
        scenario=MAIN_SCENARIO,
        observer_error=MAIN_OBSERVER_ERROR,
        output_path=figure_2,
    )

    print("Создание рисунка 3...")
    plot_article_figure_3_ablation(
        results,
        scenario=MAIN_SCENARIO,
        observer_error=MAIN_OBSERVER_ERROR,
        output_path=figure_3,
    )

    print("Создание финальной таблицы...")
    table = build_final_precision_table(
        results,
        scenario=MAIN_SCENARIO,
        observer_error=MAIN_OBSERVER_ERROR,
    )
    table.to_csv(table_1, index=False, encoding="utf-8")

    print("\nФинальные файлы сохранены:")
    print(f"  - {figure_1}")
    print(f"  - {figure_1.with_suffix('.pdf')}")
    print(f"  - {figure_2}")
    print(f"  - {figure_2.with_suffix('.pdf')}")
    print(f"  - {figure_3}")
    print(f"  - {figure_3.with_suffix('.pdf')}")
    print(f"  - {table_1}")


# -----------------------------------------------------------------------------
# Основной расчёт
# -----------------------------------------------------------------------------

def run_experiments() -> list[dict]:
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

    all_results: list[dict] = []
    start_time = time.time()

    total_configs = (
        len(AUDIT_BUDGET_VALUES)
        * len(SCENARIOS_TO_RUN)
        * len(PRIORITIZATION_MODES)
        * len(OBSERVER_ERROR_VALUES)
    )

    ablation_scenario = MAIN_SCENARIO
    ablation_observer_error = MAIN_OBSERVER_ERROR
    if INCLUDE_ABLATION:
        total_configs += len(AUDIT_BUDGET_VALUES) * len(ABLATION_MODES)

    config_counter = 0
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
                            key: round(value, 6) if isinstance(value, float) else value
                            for key, value in summary.items()
                        },
                    }
                    all_results.append(row)
                    print(
                        f"  precision={summary['audit_precision_mean']:.4f}, "
                        f"recall={summary['recall_hit_rate_mean']:.4f}, "
                        f"false_attention={summary['false_attention_rate_mean']:.4f}"
                    )

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
                    mode,
                    audit_budget,
                    ablation_scenario,
                    ablation_observer_error,
                )
                summary = run_prioritization_monte_carlo(cfg)
                row = {
                    "scenario": ablation_scenario,
                    "mode": mode,
                    "audit_budget": audit_budget,
                    "observer_error": ablation_observer_error,
                    **{
                        key: round(value, 6) if isinstance(value, float) else value
                        for key, value in summary.items()
                    },
                }
                if mode not in PRIORITIZATION_MODES:
                    all_results.append(row)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"Эксперименты завершены за {elapsed / 60:.1f} минут")
    print(f"Всего конфигураций: {total_configs}")
    print(f"Всего прогонов: {total_configs * RUNS}")
    print(f"{'=' * 80}")

    write_csv(OUT_DIR / "detailed_results_article.csv", all_results)
    write_json(OUT_DIR / "detailed_results_article.json", {"results": all_results})
    print(f"Результаты сохранены в: {OUT_DIR}")
    return all_results



def main() -> None:
    if RENDER_ONLY:
        print("=" * 80)
        print("ПЕРЕСБОРКА ФИНАЛЬНЫХ ИЛЛЮСТРАЦИЙ ПО СОХРАНЁННЫМ РЕЗУЛЬТАТАМ")
        print("=" * 80)
        results_path = OUT_DIR / "detailed_results_article.json"
        if not results_path.exists():
            raise FileNotFoundError(
                f"Не найден файл результатов: {results_path}. Сначала выполните полный эксперимент."
            )
        render_final_article_assets(load_results(results_path))
        return

    results = run_experiments()
    render_final_article_assets(results)


if __name__ == "__main__":
    main()
