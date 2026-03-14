from __future__ import annotations

from typing import Iterable

from game_ext.qtcid_repro.types import Node, QTCIDConfig
from game_ext.qtcid_repro.utils import euclidean_distance


def transmission_cost(cfg: QTCIDConfig, distance: float) -> float:
    """
    Q-TCID Eq. (2):
        C_ij = beta1 + beta2 * D_ij^theta
    """
    return cfg.beta1 + cfg.beta2 * (distance ** cfg.theta)


def link_energy(
    cfg: QTCIDConfig,
    sender: Node,
    receiver: Node,
    flow_rate: float = 1.0,
) -> float:
    """
    Упрощённая operationalization Eq. (1):
        E = rho * sum(g_ij) + sum(C_ij * g_ij)

    Для одного направленного обмена i -> j:
        E_ij = rho * g_ij + C_ij * g_ij
    """
    d = euclidean_distance(sender.pos.x, sender.pos.y, receiver.pos.x, receiver.pos.y)
    cij = transmission_cost(cfg, d)
    return cfg.rho * flow_rate + cij * flow_rate


def broadcast_energy(
    cfg: QTCIDConfig,
    sender: Node,
    receivers: list[Node],
    flow_rate: float = 1.0,
) -> float:
    total = 0.0
    for r in receivers:
        total += link_energy(cfg, sender, r, flow_rate=flow_rate)
    return total


def voting_round_energy(
    cfg: QTCIDConfig,
    target: Node,
    voters: list[Node],
    flow_rate: float = 1.0,
) -> float:
    """
    More conservative operationalization for one IDS voting round.

    We do not model full heavy bidirectional exchange as before,
    because that made nodewise Q-TCID unrealistically expensive
    compared with Wang/SOID.
    """
    if not voters:
        return 0.0

    # small communication overhead
    e_comm = 0.0
    for v in voters:
        d = euclidean_distance(target.pos.x, target.pos.y, v.pos.x, v.pos.y)
        e_comm += 0.15 * transmission_cost(cfg, d)

    # dominant cost: local detection by m voters
    e_detect = len(voters) * cfg.single_detection_cost_rate

    return e_detect + e_comm


def audit_round_energy(
    cfg: QTCIDConfig,
    target: Node,
    related_nodes: list[Node],
    flow_rate: float = 1.0,
) -> float:
    e_collect = 0.0
    for n in related_nodes:
        e_collect += link_energy(cfg, n, target, flow_rate=flow_rate)
    return cfg.audit_cost + e_collect


def apply_energy_spent(node: Node, spent: float) -> None:
    node.energy = max(0.0, node.energy - spent)


def active_energy_budget_ratio(nodes: list[Node], cfg: QTCIDConfig) -> float:
    active = [n for n in nodes if n.state != n.state.EVICTED]
    if not active:
        return 0.0
    current = sum(n.energy for n in active)
    initial = len(active) * cfg.ein
    return current / initial if initial > 0 else 0.0


def total_energy_left(nodes: list[Node]) -> float:
    return sum(n.energy for n in nodes)


def total_energy_spent(nodes: list[Node], cfg: QTCIDConfig) -> float:
    return len(nodes) * cfg.ein - total_energy_left(nodes)