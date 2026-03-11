from __future__ import annotations

from math import comb
import random


def safe_comb(n: int, k: int) -> int:
    if n < 0 or k < 0 or k > n:
        return 0
    return comb(n, k)


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def binomial_sample(rng: random.Random, n: int, p: float) -> int:
    p = clamp01(p)
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    c = 0
    for _ in range(n):
        if rng.random() < p:
            c += 1
    return c


def num_layers(beta: float) -> int:
    if abs(beta - 1.0) < 1e-9:
        return 1
    if abs(beta - 0.5) < 1e-9:
        return 2
    if abs(beta - (1.0 / 3.0)) < 1e-9:
        return 3
    raise ValueError("Supported beta values: 1, 1/2, 1/3")