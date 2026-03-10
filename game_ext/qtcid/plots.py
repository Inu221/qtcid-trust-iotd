from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt


def plot_curve(
    x: Iterable[float],
    y: Iterable[float],
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
) -> None:
    x_list: List[float] = list(x)
    y_list: List[float] = list(y)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(x_list, y_list, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()