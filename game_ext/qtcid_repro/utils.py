from __future__ import annotations

import math
import random
import statistics
from typing import Iterable, List, Sequence, Tuple


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def majority_threshold(m: int) -> int:
    return m // 2 + 1


def bernoulli(rng: random.Random, p: float) -> bool:
    return rng.random() < clamp01(p)


def binomial_sample(rng: random.Random, n: int, p: float) -> int:
    p = clamp01(p)
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n

    count = 0
    for _ in range(n):
        if rng.random() < p:
            count += 1
    return count


def random_choice_without_replacement(
    rng: random.Random,
    items: Sequence[int],
    k: int,
) -> list[int]:
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    return rng.sample(list(items), k)


def summarize_mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0