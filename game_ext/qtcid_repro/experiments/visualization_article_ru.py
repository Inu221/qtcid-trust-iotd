"""
Модуль визуализации для статьи с полной русификацией.

Создаёт профессиональные графики с использованием seaborn:
1. Line plots с confidence bands
2. Heatmap улучшения history vs current
3. Grouped bar chart для главного сценария
4. Ablation plot
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Настройка стиля
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Цвета для методов
COLORS = {
    "random": "#ff7f0e",
    "current_only": "#2ca02c",
    "history_based": "#d62728",
    "history_no_stability": "#9467bd",
}

# Русские метки для методов
LABELS_RU = {
    "random": "Случайный выбор",
    "current_only": "Только текущие наблюдения",
    "history_based": "С историей (предлагаемый)",
    "history_no_stability": "С историей без стабильности",
}

# Русские метки для сценариев
SCENARIO_LABELS_RU = {
    "low_noise_persistent": "Низкий шум, устойчивые атаки",
    "medium_noise_mixed": "Средний шум, смешанные атаки",
    "high_noise_intermittent": "Высокий шум, эпизодические атаки",
}


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> tuple:
    """Вычислить доверительный интервал"""
    if len(values) < 2:
        return 0.0, 0.0

    mean = np.mean(values)
    sem = stats.sem(values)
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    return mean - ci, mean + ci


def prepare_data_with_ci(results: List[dict], group_keys: List[str], value_key: str) -> pd.DataFrame:
    """
    Подготовить данные с доверительными интервалами.

    Группирует результаты по group_keys и вычисляет mean, ci_low, ci_high для value_key.
    """
    df = pd.DataFrame(results)

    # Группировать и вычислить статистику
    grouped = df.groupby(group_keys)[value_key].apply(list).reset_index()

    stats_data = []
    for _, row in grouped.iterrows():
        values = row[value_key]
        mean_val = np.mean(values)
        ci_low, ci_high = compute_confidence_interval(values)

        record = {k: row[k] for k in group_keys}
        record.update({
            'mean': mean_val,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'std': np.std(values),
        })
        stats_data.append(record)

    return pd.DataFrame(stats_data)


# -----------------------------------------------------------------------------
# 1. Line plots с confidence bands
# -----------------------------------------------------------------------------

def plot_line_with_ci(
    results: List[dict],
    x_key: str,
    value_key: str,
    group_key: str,
    filter_dict: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    figsize: tuple = (10, 6),
):
    """
    Line plot с confidence bands.

    Parameters:
    - results: список словарей с результатами
    - x_key: ключ для оси X (например, "audit_budget")
    - value_key: ключ для значения на оси Y
    - group_key: ключ для группировки линий (например, "mode")
    - filter_dict: фильтры для subset (например, {"scenario": "medium_noise_mixed"})
    - title, xlabel, ylabel: русские подписи
    - output_path: путь для сохранения
    """
    # Фильтрация
    subset = [r for r in results if all(r.get(k) == v for k, v in filter_dict.items())]

    if not subset:
        print(f"Нет данных для {filter_dict}")
        return

    # Подготовить данные с CI
    df_stats = prepare_data_with_ci(subset, [x_key, group_key], value_key)

    fig, ax = plt.subplots(figsize=figsize)

    groups = df_stats[group_key].unique()
    for group in groups:
        group_data = df_stats[df_stats[group_key] == group].sort_values(x_key)

        x = group_data[x_key].values
        y = group_data['mean'].values
        ci_low = group_data['ci_low'].values
        ci_high = group_data['ci_high'].values

        color = COLORS.get(group, "#333333")
        label = LABELS_RU.get(group, group)

        # Линия
        ax.plot(x, y, marker='o', linewidth=2.5, color=color, label=label, markersize=7)

        # Confidence band
        ax.fill_between(x, ci_low, ci_high, color=color, alpha=0.2)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=15)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# 2. Heatmap улучшения
# -----------------------------------------------------------------------------

def plot_improvement_heatmap(
    results: List[dict],
    metric_key: str,
    title: str,
    output_path: Path,
    filter_dict: dict = None,
    figsize: tuple = (10, 7),
):
    """
    Heatmap улучшения history_based относительно current_only.

    По осям: audit_budget × observer_error
    Значения: относительное улучшение (history - current) / current * 100%
    """
    subset = results
    if filter_dict:
        subset = [r for r in subset if all(r.get(k) == v for k, v in filter_dict.items())]

    if not subset:
        print(f"Нет данных для heatmap: {filter_dict}")
        return

    df = pd.DataFrame(subset)

    # Вычислить mean для каждой комбинации
    df_agg = df.groupby(['audit_budget', 'observer_error', 'mode'])[metric_key].mean().reset_index()

    # Создать таблицу improvement
    budgets = sorted(df_agg['audit_budget'].unique())
    errors = sorted(df_agg['observer_error'].unique())

    improvement_matrix = np.zeros((len(errors), len(budgets)))

    for i, err in enumerate(errors):
        for j, budget in enumerate(budgets):
            current_val = df_agg[(df_agg['audit_budget'] == budget) &
                                  (df_agg['observer_error'] == err) &
                                  (df_agg['mode'] == 'current_only')][metric_key].values
            history_val = df_agg[(df_agg['audit_budget'] == budget) &
                                  (df_agg['observer_error'] == err) &
                                  (df_agg['mode'] == 'history_based')][metric_key].values

            if len(current_val) > 0 and len(history_val) > 0 and current_val[0] > 0:
                improvement = (history_val[0] - current_val[0]) / current_val[0] * 100
                improvement_matrix[i, j] = improvement

    fig, ax = plt.subplots(figsize=figsize)

    # Heatmap
    im = ax.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=30)

    # Подписи осей
    ax.set_xticks(np.arange(len(budgets)))
    ax.set_yticks(np.arange(len(errors)))
    ax.set_xticklabels(budgets)
    ax.set_yticklabels([f"{e:.2f}" for e in errors])

    ax.set_xlabel("Бюджет системной проверки (узлов за цикл)", fontsize=12)
    ax.set_ylabel("Уровень ошибки наблюдателя", fontsize=12)
    ax.set_title(title, fontsize=13, pad=15)

    # Аннотации
    for i in range(len(errors)):
        for j in range(len(budgets)):
            val = improvement_matrix[i, j]
            text = ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                          color="black" if abs(val) < 15 else "white", fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Улучшение (%)", rotation=270, labelpad=20, fontsize=11)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# 3. Grouped bar chart
# -----------------------------------------------------------------------------

def plot_grouped_bar_chart(
    results: List[dict],
    metrics: List[tuple],  # [(metric_key, ylabel), ...]
    budgets: List[int],
    filter_dict: dict,
    title: str,
    output_path: Path,
    figsize: tuple = (14, 10),
):
    """
    Grouped bar chart для нескольких метрик при разных бюджетах.

    Parameters:
    - metrics: список кортежей (metric_key, ylabel)
    - budgets: список значений бюджета для отображения
    """
    subset = [r for r in results if all(r.get(k) == v for k, v in filter_dict.items())]

    if not subset:
        print(f"Нет данных для grouped bar chart: {filter_dict}")
        return

    df = pd.DataFrame(subset)
    df = df[df['audit_budget'].isin(budgets)]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    modes = ['random', 'current_only', 'history_based']
    x = np.arange(len(budgets))
    width = 0.25

    for idx, (metric_key, ylabel) in enumerate(metrics):
        if idx >= len(axes):
            break

        ax = axes[idx]

        for i, mode in enumerate(modes):
            mode_data = df[df['mode'] == mode].groupby('audit_budget')[metric_key].mean().reindex(budgets, fill_value=0)
            ax.bar(x + i * width, mode_data.values, width, label=LABELS_RU[mode], color=COLORS[mode])

        ax.set_xlabel("Бюджет (узлов за цикл)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels(budgets)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Удалить пустые подграфики
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# 4. Ablation plot
# -----------------------------------------------------------------------------

def plot_ablation_study(
    results: List[dict],
    metric_key: str,
    filter_dict: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    figsize: tuple = (10, 6),
):
    """
    Ablation plot: current_only vs history_no_stability vs history_based.
    """
    subset = [r for r in results if all(r.get(k) == v for k, v in filter_dict.items())]

    if not subset:
        print(f"Нет данных для ablation plot: {filter_dict}")
        return

    ablation_modes = ['current_only', 'history_no_stability', 'history_based']
    subset = [r for r in subset if r['mode'] in ablation_modes]

    df_stats = prepare_data_with_ci(subset, ['audit_budget', 'mode'], metric_key)

    fig, ax = plt.subplots(figsize=figsize)

    for mode in ablation_modes:
        mode_data = df_stats[df_stats['mode'] == mode].sort_values('audit_budget')

        x = mode_data['audit_budget'].values
        y = mode_data['mean'].values
        ci_low = mode_data['ci_low'].values
        ci_high = mode_data['ci_high'].values

        color = COLORS.get(mode, "#333333")
        label = LABELS_RU.get(mode, mode)

        ax.plot(x, y, marker='o', linewidth=2.5, color=color, label=label, markersize=7)
        ax.fill_between(x, ci_low, ci_high, color=color, alpha=0.2)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=15)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# 5. Сравнительная панель для главного сценария
# -----------------------------------------------------------------------------

def plot_main_scenario_panel(
    results: List[dict],
    scenario: str,
    observer_error: float,
    output_path: Path,
):
    """
    Главная панель для статьи: 6 ключевых метрик с confidence bands.
    """
    filter_dict = {"scenario": scenario, "observer_error": observer_error}

    metrics = [
        ("audit_precision_mean", "Точность отбора узлов на проверку"),
        ("recall_hit_rate_mean", "Полнота выявления проблемных узлов"),
        ("intermittent_detection_rate_mean", "Доля выявления эпизодических атак"),
        ("false_attention_rate_mean", "Доля ложного внимания к шумным узлам"),
        ("cumulative_residual_risk_mean", "Накопленный остаточный риск"),
        ("mean_cycles_to_verify_intermittent", "Среднее число циклов до обнаружения"),
    ]

    subset = [r for r in results if all(r.get(k) == v for k, v in filter_dict.items())]

    if not subset:
        print(f"Нет данных для главной панели: {filter_dict}")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    modes = ['random', 'current_only', 'history_based']

    for idx, (metric_key, ylabel) in enumerate(metrics):
        ax = axes[idx]

        df_stats = prepare_data_with_ci(subset, ['audit_budget', 'mode'], metric_key)

        for mode in modes:
            mode_data = df_stats[df_stats['mode'] == mode].sort_values('audit_budget')

            if len(mode_data) == 0:
                continue

            x = mode_data['audit_budget'].values
            y = mode_data['mean'].values
            ci_low = mode_data['ci_low'].values
            ci_high = mode_data['ci_high'].values

            color = COLORS[mode]
            label = LABELS_RU[mode]

            ax.plot(x, y, marker='o', linewidth=2, color=color, label=label, markersize=6)
            ax.fill_between(x, ci_low, ci_high, color=color, alpha=0.15)

        ax.set_xlabel("Бюджет (узлов за цикл)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11)

        if idx == 0:
            ax.legend(fontsize=8, loc='best', frameon=True)

        ax.grid(True, alpha=0.3, linestyle='--')

    scenario_title = SCENARIO_LABELS_RU.get(scenario, scenario)
    fig.suptitle(f"Главный сценарий: {scenario_title} (ошибка наблюдателя = {observer_error:.2f})",
                 fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
