from __future__ import annotations

from math import comb

from game_ext.qtcid_repro.utils import clamp01, safe_div


def safe_comb(n: int, k: int) -> int:
    if n < 0 or k < 0 or k > n:
        return 0
    return comb(n, k)


def majority_threshold(m: int) -> int:
    return m // 2 + 1


def voting_error_probability_basic(
    n_good: int,
    n_bad: int,
    m: int,
    omega: float,
) -> float:
    total = n_good + n_bad
    if m <= 0 or total < m:
        return 0.0

    mmaj = majority_threshold(m)
    denom = safe_comb(total, m)
    if denom == 0:
        return 0.0

    # Case 1: majority of selected voters are bad
    s0 = 0.0
    max_i_major = min(n_bad - mmaj, m - mmaj) if n_bad >= mmaj else -1
    for i in range(0, max_i_major + 1):
        bad_cnt = mmaj + i
        good_cnt = m - bad_cnt
        s0 += safe_div(
            safe_comb(n_bad, bad_cnt) * safe_comb(n_good, good_cnt),
            denom,
        )

    # Case 2: minority of selected voters are bad, but enough good voters vote incorrectly
    s1 = 0.0
    max_j_bad_minor = min(n_bad, m - mmaj)
    for j in range(0, max_j_bad_minor + 1):
        sigma = 0.0
        for k in range(mmaj - j, m - j + 1):
            if k > n_good:
                continue
            sigma += (
                safe_comb(n_good, k)
                * (omega ** k)
                * safe_comb(n_good - k, m - j - k)
                * ((1.0 - omega) ** (m - j - k))
            )
        s1 += safe_div(safe_comb(n_bad, j) * sigma, denom)

    return clamp01(s0 + s1)


def _wang_ids_error_probability_fixed_active(
    n_good: int,
    n_bad: int,
    m: int,
    n_bad_active: int,
    omega: float,
) -> float:
    total = n_good + n_bad
    if total < m or m <= 0:
        return 0.0

    n_bad_active = max(0, min(n_bad, n_bad_active))
    n_bad_inactive = n_bad - n_bad_active
    n_honest_pool = n_good + n_bad_inactive
    mmaj = majority_threshold(m)

    denom = safe_comb(total, m)
    if denom == 0:
        return 0.0

    term1 = 0.0
    for i in range(0, m - mmaj + 1):
        a = mmaj + i
        b = m - a
        term1 += safe_div(
            safe_comb(n_bad_active, a) * safe_comb(n_honest_pool, b),
            denom,
        )

    term2 = 0.0
    for i in range(0, m - mmaj + 1):
        choose_active_bad = safe_comb(n_bad_active, i)
        inner = 0.0

        for j in range(mmaj - i, m - i + 1):
            choose_wrong = safe_comb(n_honest_pool, j)
            choose_right = safe_comb(n_honest_pool - j, m - i - j)
            inner += (
                choose_wrong
                * (omega ** j)
                * choose_right
                * ((1.0 - omega) ** (m - i - j))
            )

        term2 += safe_div(choose_active_bad * inner, denom)

    return clamp01(term1 + term2)


def wang_ids_error_probability(
    n_good: int,
    n_bad: int,
    m: int,
    pa: float,
    omega: float,
) -> float:
    """
    Wang Eq. (6) with analytical averaging over the number of active bad nodes.

    Instead of:
        n_bad_active = round(pa * n_bad)

    we average over:
        K ~ Binomial(n_bad, pa)

    This better matches Wang's closed-form / random attack behavior.
    """
    total = n_good + n_bad
    if total < m or m <= 0:
        return 0.0

    if n_bad <= 0:
        return _wang_ids_error_probability_fixed_active(
            n_good=n_good,
            n_bad=n_bad,
            m=m,
            n_bad_active=0,
            omega=omega,
        )

    pa = clamp01(pa)

    result = 0.0
    for k in range(0, n_bad + 1):
        pk = safe_comb(n_bad, k) * (pa ** k) * ((1.0 - pa) ** (n_bad - k))
        result += pk * _wang_ids_error_probability_fixed_active(
            n_good=n_good,
            n_bad=n_bad,
            m=m,
            n_bad_active=k,
            omega=omega,
        )

    return clamp01(result)