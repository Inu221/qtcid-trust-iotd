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

    # i = число заведомо "плохих" голосующих
    # j = число "хороших", ошибочно проголосовавших против истины
    p_err = 0.0

    for i in range(0, min(n_bad, m) + 1):
        choose_bad = safe_comb(n_bad, i)
        remain = m - i

        for j in range(0, remain + 1):
            if i + j < mmaj:
                continue

            choose_wrong_honest = safe_comb(n_good, j)
            choose_right_honest = safe_comb(n_good - j, remain - j)

            numerator = (
                choose_bad
                * choose_wrong_honest
                * choose_right_honest
                * (omega ** j)
                * ((1.0 - omega) ** (remain - j))
            )
            p_err += safe_div(numerator, denom)

    return clamp01(p_err)


def wang_ids_error_probability(
    n_good: int,
    n_bad: int,
    m: int,
    pa: float,
    omega: float,
) -> float:
    """
    Wang Eq. (6),

    n_bad_active = Pa * Nb
    n_bad_inactive = (1 - Pa) * Nb
    omega = Hpfp for P_fp^IDS
    omega = Hpfn for P_fn^IDS
    """
    total = n_good + n_bad
    if total < m or m <= 0:
        return 0.0

    n_bad_active = max(0, min(n_bad, round(pa * n_bad)))
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