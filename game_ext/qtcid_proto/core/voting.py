from __future__ import annotations

from .utils import safe_comb, safe_div, clamp01


def wang_ids_error_probability(
    n_good: int,
    n_bad: int,
    m: int,
    pa: float,
    omega: float,
) -> float:
    total = n_good + n_bad
    if total < m or m <= 0:
        return 0.0

    n_bad_active = max(0, min(n_bad, round(pa * n_bad)))
    n_bad_inactive = n_bad - n_bad_active
    n_honest_pool = n_good + n_bad_inactive
    m_major = m // 2 + 1
    denom = safe_comb(total, m)

    if denom == 0:
        return 0.0

    term1 = 0.0
    for i in range(0, m - m_major + 1):
        a = m_major + i
        b = m - a
        term1 += safe_div(
            safe_comb(n_bad_active, a) * safe_comb(n_honest_pool, b),
            denom,
        )

    term2 = 0.0
    for i in range(0, m - m_major + 1):
        choose_active_bad = safe_comb(n_bad_active, i)
        inner = 0.0
        for j in range(m_major - i, m - i + 1):
            choose_wrong = safe_comb(n_honest_pool, j)
            choose_right = safe_comb(n_honest_pool - j, m - i - j)
            inner += choose_wrong * (omega ** j) * choose_right * ((1.0 - omega) ** (m - i - j))
        term2 += safe_div(choose_active_bad * inner, denom)

    return clamp01(term1 + term2)