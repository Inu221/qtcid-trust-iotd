"""
Финальная визуализация для статьи по приоритизации системной проверки.

Модуль формирует три основных рисунка и финальную сводную таблицу для статьи.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

COLORS = {
    "history_based": "#2ca02c",
    "current_only": "#ff7f0e",
    "random": "#7f7f7f",
    "history_no_stability": "#1f77b4",
}

LABELS_RU = {
    "history_based": "Предлагаемый метод",
    "current_only": "Только текущие
наблюдения",
    "random": "Случайный выбор",
    "history_no_stability": "С историей без компонента
стабильности",
}

MAIN_METHOD_ORDER = ["history_based", "current_only", "random"]
ABLATION_METHOD_ORDER = ["current_only", "history_no_stability", "history_based"]

X_LABEL_BUDGET = "Бюджет системной проверки,
узлов за цикл"
Y_LABEL_OBSERVER_ERROR = "Уровень ошибки наблюдателя, %"
Y_LABEL_IMPROVEMENT = "Улучшение точности отбора, %"
Y_LABEL_PRECISION = "Точность отбора узлов
на проверку"
Y_LABEL_RECALL = "Полнота выявления
проблемных узлов"
Y_LABEL_FALSE_ATTENTION = "Ложные проверки корректных
шумных узлов, доля"


def _as_dataframe(results: Iterable[dict]) -> pd.DataFrame:
    return pd.DataFrame(list(results)).copy()



def _filter_dataframe(df: pd.DataFrame, filter_dict: dict | None = None) -> pd.DataFrame:
    if not filter_dict:
        return df.copy()

    subset = df.copy()
    for key, value in filter_dict.items():
        subset = subset[subset[key] == value]
    return subset



def _aggregate_metric(df: pd.DataFrame, group_keys: list[str], metric_key: str) -> pd.DataFrame:
    return df.groupby(group_keys, as_index=False)[metric_key].mean()



def _metric_ylim(
    values: Iterable[float],
    lower_bound: float = 0.0,
    upper_bound: float | None = 1.0,
) -> tuple[float, float]:
    clean_values = [float(value) for value in values if pd.notna(value)]
    if not clean_values:
        if upper_bound is None:
            return lower_bound, lower_bound + 1.0
        return lower_bound, upper_bound

    min_value = min(clean_values)
    max_value = max(clean_values)
    span = max(max_value - min_value, 0.06)
    y_min = min_value - 0.15 * span
    y_max = max_value + 0.15 * span

    if lower_bound is not None:
        y_min = max(lower_bound, y_min)
    if upper_bound is not None:
        y_max = min(upper_bound, y_max)

    if y_max <= y_min:
        y_max = y_min + 0.1
    return y_min, y_max



def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.25)
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", pad=4)



def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.5,
        1.08,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="normal",
    )



def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)



def plot_article_figure_1_heatmap(
    results: Iterable[dict],
    scenario: str,
    output_path: Path,
) -> None:
    """Рисунок 1: heatmap улучшения точности отбора предлагаемого метода."""
    df = _filter_dataframe(_as_dataframe(results), {"scenario": scenario})
    if df.empty:
        print(f"Нет данных для рисунка 1: scenario={scenario}")
        return

    grouped = _aggregate_metric(
        df,
        ["audit_budget", "observer_error", "mode"],
        "audit_precision_mean",
    )
    budgets = sorted(grouped["audit_budget"].unique())
    errors = sorted(grouped["observer_error"].unique())

    heatmap = pd.DataFrame(
        index=[round(error * 100) for error in errors],
        columns=budgets,
        dtype=float,
    )
    for budget in budgets:
        for error in errors:
            current_values = grouped[
                (grouped["audit_budget"] == budget)
                & (grouped["observer_error"] == error)
                & (grouped["mode"] == "current_only")
            ]["audit_precision_mean"].values
            proposed_values = grouped[
                (grouped["audit_budget"] == budget)
                & (grouped["observer_error"] == error)
                & (grouped["mode"] == "history_based")
            ]["audit_precision_mean"].values

            if len(current_values) == 0 or len(proposed_values) == 0 or current_values[0] <= 0:
                heatmap.loc[round(error * 100), budget] = np.nan
                continue

            heatmap.loc[round(error * 100), budget] = (
                100.0 * (proposed_values[0] - current_values[0]) / current_values[0]
            )

    valid_values = heatmap.to_numpy(dtype=float)
    valid_values = valid_values[~np.isnan(valid_values)]
    if valid_values.size == 0:
        print("Нет значений для рисунка 1")
        return

    max_abs = float(np.max(np.abs(valid_values)))
    color_limit = max(10.0, math.ceil(max_abs / 10.0) * 10.0)
    annotations = heatmap.apply(
        lambda column: column.map(lambda value: "" if pd.isna(value) else f"{value:+.1f}")
    )

    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    cmap = sns.diverging_palette(15, 130, as_cmap=True)
    sns.heatmap(
        heatmap,
        ax=ax,
        cmap=cmap,
        center=0.0,
        vmin=-color_limit,
        vmax=color_limit,
        annot=annotations,
        fmt="",
        linewidths=1.0,
        linecolor="white",
        cbar_kws={"label": Y_LABEL_IMPROVEMENT},
    )

    ax.set_xlabel(X_LABEL_BUDGET, labelpad=10)
    ax.set_ylabel(Y_LABEL_OBSERVER_ERROR, labelpad=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    cbar.set_label(Y_LABEL_IMPROVEMENT, rotation=270, labelpad=18)
    fig.tight_layout()
    _save_figure(fig, output_path)



def plot_article_figure_2_main_panel(
    results: Iterable[dict],
    scenario: str,
    observer_error: float,
    output_path: Path,
) -> None:
    """Рисунок 2: составной рисунок для главного сценария."""
    df = _filter_dataframe(
        _as_dataframe(results),
        {"scenario": scenario, "observer_error": observer_error},
    )
    if df.empty:
        print(f"Нет данных для рисунка 2: scenario={scenario}, observer_error={observer_error}")
        return

    metrics = [
        ("audit_precision_mean", Y_LABEL_PRECISION, "а)"),
        ("recall_hit_rate_mean", Y_LABEL_RECALL, "б)"),
        ("false_attention_rate_mean", Y_LABEL_FALSE_ATTENTION, "в)"),
    ]
    budgets = sorted(df["audit_budget"].unique())

    fig = plt.figure(figsize=(13.4, 9.8))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.08])
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 1:3]),
    ]

    for ax, (metric_key, ylabel, panel_label) in zip(axes, metrics):
        grouped = _aggregate_metric(df, ["audit_budget", "mode"], metric_key)
        for mode in MAIN_METHOD_ORDER:
            mode_data = grouped[grouped["mode"] == mode].sort_values("audit_budget")
            if mode_data.empty:
                continue

            ax.plot(
                mode_data["audit_budget"],
                mode_data[metric_key],
                color=COLORS[mode],
                label=LABELS_RU[mode],
                linewidth=2.5,
                marker="o",
                markersize=6.5,
                markerfacecolor="white",
                markeredgewidth=1.5,
            )

        ax.set_xlabel(X_LABEL_BUDGET, labelpad=10)
        ax.set_ylabel(ylabel, labelpad=12)
        ax.set_xticks(budgets)
        ax.set_xlim(min(budgets) - 0.3, max(budgets) + 0.3)
        ax.set_ylim(
            *_metric_ylim(grouped[metric_key].values, lower_bound=0.0, upper_bound=1.0)
        )
        _style_axis(ax)
        _add_panel_label(ax, panel_label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.96,
        bottom=0.14,
        wspace=0.42,
        hspace=0.68,
    )
    _save_figure(fig, output_path)



def plot_article_figure_3_ablation(
    results: Iterable[dict],
    scenario: str,
    observer_error: float,
    output_path: Path,
) -> None:
    """Рисунок 3: вклад компонентов истории в точность отбора."""
    df = _filter_dataframe(
        _as_dataframe(results),
        {"scenario": scenario, "observer_error": observer_error},
    )
    if df.empty:
        print(f"Нет данных для рисунка 3: scenario={scenario}, observer_error={observer_error}")
        return

    df = df[df["mode"].isin(ABLATION_METHOD_ORDER)]
    if df.empty:
        print("Нет ablation-данных для рисунка 3")
        return

    grouped = _aggregate_metric(df, ["audit_budget", "mode"], "audit_precision_mean")
    budgets = sorted(grouped["audit_budget"].unique())

    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    for mode in ABLATION_METHOD_ORDER:
        mode_data = grouped[grouped["mode"] == mode].sort_values("audit_budget")
        if mode_data.empty:
            continue

        ax.plot(
            mode_data["audit_budget"],
            mode_data["audit_precision_mean"],
            color=COLORS[mode],
            label=LABELS_RU[mode],
            linewidth=2.5,
            marker="o",
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.5,
        )

    ax.set_xlabel(X_LABEL_BUDGET, labelpad=10)
    ax.set_ylabel(Y_LABEL_PRECISION, labelpad=12)
    ax.set_xticks(budgets)
    ax.set_xlim(min(budgets) - 0.3, max(budgets) + 0.3)
    ax.set_ylim(*_metric_ylim(grouped["audit_precision_mean"].values, lower_bound=0.0, upper_bound=1.0))
    ax.legend(loc="upper left", frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    _save_figure(fig, output_path)



def build_final_precision_table(
    results: Iterable[dict],
    scenario: str,
    observer_error: float,
) -> pd.DataFrame:
    """Финальная таблица для статьи: current-only vs proposed."""
    df = _filter_dataframe(
        _as_dataframe(results),
        {"scenario": scenario, "observer_error": observer_error},
    )
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Бюджет системной проверки, узлов за цикл",
                "Точность: только текущие наблюдения",
                "Точность: предлагаемый метод",
                "Улучшение точности отбора, %",
            ]
        )

    grouped = _aggregate_metric(df, ["audit_budget", "mode"], "audit_precision_mean")
    rows = []
    for budget in sorted(grouped["audit_budget"].unique()):
        current_values = grouped[
            (grouped["audit_budget"] == budget) & (grouped["mode"] == "current_only")
        ]["audit_precision_mean"].values
        proposed_values = grouped[
            (grouped["audit_budget"] == budget) & (grouped["mode"] == "history_based")
        ]["audit_precision_mean"].values

        if len(current_values) == 0 or len(proposed_values) == 0:
            continue

        current_value = float(current_values[0])
        proposed_value = float(proposed_values[0])
        improvement = 0.0
        if current_value > 0:
            improvement = 100.0 * (proposed_value - current_value) / current_value

        rows.append({
            "Бюджет системной проверки, узлов за цикл": int(budget),
            "Точность: только текущие наблюдения": f"{current_value:.3f}",
            "Точность: предлагаемый метод": f"{proposed_value:.3f}",
            "Улучшение точности отбора, %": f"{improvement:+.1f}",
        })

    return pd.DataFrame(rows)
