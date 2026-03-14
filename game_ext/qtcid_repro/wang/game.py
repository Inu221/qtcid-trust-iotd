from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WangGameConfig:
    """
    Wang attack-defense game parameters.

    beta:
        life quota decay parameter
        supported values: 1, 1/2, 1/3
    pc:
        auditing probability
    pa:
        attack probability
    """
    beta: float = 1.0
    pc: float = 0.5
    pa: float = 0.5


def supported_beta(beta: float) -> bool:
    return abs(beta - 1.0) < 1e-9 or abs(beta - 0.5) < 1e-9 or abs(beta - (1.0 / 3.0)) < 1e-9


def num_life_layers(beta: float) -> int:
    if abs(beta - 1.0) < 1e-9:
        return 1
    if abs(beta - 0.5) < 1e-9:
        return 2
    if abs(beta - (1.0 / 3.0)) < 1e-9:
        return 3
    raise ValueError("Supported beta values are 1, 1/2, 1/3")


def pmin_c(beta: float) -> float:
    """
    Wang, from condition (5) with life quota payoff parameterization:
        Pc_min = 1 / (1 + beta)
    """
    return 1.0 / (1.0 + beta)


def payoff_params_from_life_quota(beta: float) -> tuple[float, float, float, float]:
    """
    Returns:
        (L_a_c, L_na_c, G_a_nc, G_na_nc)

    As defined in Wang section IV.A:
        L_a^c   = beta
        L_na^c  = 0
        G_a^nc  = 1
        G_na^nc = 0
    """
    return beta, 0.0, 1.0, 0.0


def theorem1_attack_discouraged(pc: float, beta: float) -> bool:
    """
    Condition (4)/(5) instantiated with Wang life quota parameterization.
    """
    l_a_c, l_na_c, g_a_nc, g_na_nc = payoff_params_from_life_quota(beta)
    lhs = pc * (l_a_c - l_na_c)
    rhs = (1.0 - pc) * (g_a_nc - g_na_nc)
    return lhs >= rhs


def bad_node_counts(nb_total: int, pa: float) -> tuple[int, int]:
    """
    Returns:
        (n_bad_active, n_bad_inactive)
    """
    n_active = round(pa * nb_total)
    n_active = max(0, min(nb_total, n_active))
    n_inactive = nb_total - n_active
    return n_active, n_inactive


def layer_life_value(layer_idx: int, beta: float) -> float:
    """
    Human-readable life quota value for layer index.

    layer_idx = 0 is top layer.
    Example:
        beta=1     -> 1 layer: [1]
        beta=1/2   -> 2 layers: [1, 1/2]
        beta=1/3   -> 3 layers: [1, 2/3, 1/3]
    """
    n = num_life_layers(beta)
    return (n - layer_idx) / n


def penalize_bad_node_from_layer(layer_idx: int, beta: float) -> tuple[bool, int | None]:
    """
    Applies one audited-attack penalty.

    Returns:
        (evicted, next_layer_idx)

    If evicted is True, next_layer_idx is None.
    Otherwise the node moves to the next lower life-quota layer.
    """
    n_layers = num_life_layers(beta)

    if layer_idx >= n_layers - 1:
        return True, None

    return False, layer_idx + 1