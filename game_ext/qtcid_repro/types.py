from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class NodeState(str, Enum):
    GOOD = "GOOD"
    BAD = "BAD"
    EVICTED = "EVICTED"


@dataclass
class Position2D:
    x: float
    y: float


@dataclass
class Node:
    node_id: int
    pos: Position2D
    state: NodeState = NodeState.GOOD
    energy: float = 2.0

    # История голосования:
    # key = target_node_id, value = список прошлых голосов:
    # +1 = голосовал "good", -1 = голосовал "bad"
    vote_history: Dict[int, List[int]] = field(default_factory=dict)

    # История расхождений с аудитом / коллективным решением
    mismatch_count: int = 0

    # Число раз, когда узел был выбран голосующим
    participated_votes: int = 0

    # Число раз, когда узел был целью голосования
    targeted_count: int = 0

    def is_active(self, emin: float) -> bool:
        return self.state != NodeState.EVICTED and self.energy > emin

    def remember_vote(self, target_id: int, vote_value: int) -> None:
        if target_id not in self.vote_history:
            self.vote_history[target_id] = []
        self.vote_history[target_id].append(vote_value)

    def historical_bias_toward(self, target_id: int) -> float:
        """
        Средняя направленность прошлых голосов:
        +1 -> чаще голосовал "good"
        -1 -> чаще голосовал "bad"
        0 -> нет истории / нейтрально
        """
        values = self.vote_history.get(target_id, [])
        if not values:
            return 0.0
        return sum(values) / len(values)


@dataclass
class VotingResult:
    target_id: int
    target_state_before_vote: NodeState
    voter_ids: List[int]
    votes_for_good: int
    votes_for_bad: int
    majority_label_good: bool
    audited: bool = False
    audit_detected_attack: bool = False


@dataclass
class QLearningConfig:
    alpha: float
    gamma: float
    epsilon: float
    epsilon_decay: float = 0.997
    epsilon_min: float = 0.05


@dataclass
class QTCIDConfig:
    # Геометрия и сеть
    n_nodes: int = 128
    region_length: float = 1000.0
    region_width: float = 1000.0
    communication_range: float = 250.0

    # IDS / атаки
    tids: int = 200
    m_voters: int = 5
    pa: float = 0.5
    pcap_rate: float = 1 / 3600
    hpfp: float = 0.05
    hpfn: float = 0.05

    # Энергия
    ein: float = 2.0
    emin: float = 0.05
    audit_cost: float = 0.05

    # Формула из статьи Q-TCID:
    # E = rho * sum(g_ij) + sum(C_ij * g_ij)
    rho: float = 0.01
    beta1: float = 0.01
    beta2: float = 0.001
    theta: float = 2.0

    # Эквивалент параметра z из таблицы Q-TCID
    single_detection_cost_rate: float = 1 / 20

    # Q-learning
    host_q: QLearningConfig = field(
        default_factory=lambda: QLearningConfig(alpha=0.6, gamma=0.9, epsilon=0.5)
    )
    system_q: QLearningConfig = field(
        default_factory=lambda: QLearningConfig(alpha=0.4, gamma=0.9, epsilon=0.5)
    )

    # Награда system-level audit
    reward_b: float = 1.0

    # Симуляция
    n_mc_runs: int = 1000
    max_time: int = 20000
    seed: int = 42


@dataclass
class RunStats:
    mttf: float
    byzantine_failed: bool
    energy_failed: bool

    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    votes_total: int = 0
    audits_total: int = 0

    energy_spent_total: float = 0.0
    energy_spent_voting: float = 0.0
    energy_spent_audit: float = 0.0

    good_left: int = 0
    bad_left: int = 0
    evicted_left: int = 0

    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    def fpr(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0

    def fnr(self) -> float:
        denom = self.fn + self.tp
        return self.fn / denom if denom > 0 else 0.0


@dataclass
class SummaryStats:
    runs: int
    mttf_mean: float
    mttf_std: float
    accuracy_mean: float
    precision_mean: float
    recall_mean: float
    fpr_mean: float
    fnr_mean: float
    energy_spent_mean: float
    energy_voting_mean: float
    energy_audit_mean: float
    audits_mean: float
    byzantine_fail_ratio: float
    energy_fail_ratio: float
    good_left_mean: float
    bad_left_mean: float
    evicted_left_mean: float