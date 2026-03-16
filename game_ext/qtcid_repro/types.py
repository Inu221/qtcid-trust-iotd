from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


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

    vote_history: Dict[int, List[int]] = field(default_factory=dict)
    mismatch_count: int = 0
    participated_votes: int = 0
    targeted_count: int = 0

    def is_active(self, emin: float) -> bool:
        return self.state != NodeState.EVICTED and self.energy > emin

    def remember_vote(self, target_id: int, vote_value: int) -> None:
        if target_id not in self.vote_history:
            self.vote_history[target_id] = []
        self.vote_history[target_id].append(vote_value)

    def historical_bias_toward(self, target_id: int) -> float:
        values = self.vote_history.get(target_id, [])
        if not values:
            return 0.0
        return sum(values) / len(values)


@dataclass
class QLearningConfig:
    alpha: float
    gamma: float
    epsilon: float
    epsilon_decay: float = 0.997
    epsilon_min: float = 0.05


@dataclass
class QTCIDConfig:
    # topology / deployment
    n_nodes: int = 128
    region_length: float = 1000.0
    region_width: float = 1000.0
    communication_range: float = 250.0

    # system / attacks
    tids: int = 200
    m_voters: int = 5
    pa: float = 0.5
    pcap_rate: float = 1 / 3600
    hpfp: float = 0.05
    hpfn: float = 0.05

    # energy
    ein: float = 2.0
    emin: float = 0.05
    audit_cost: float = 0.05

    rho: float = 0.01
    beta1: float = 0.01
    beta2: float = 0.001
    theta: float = 2.0

    # z from article
    single_detection_cost_rate: float = 1 / 20

    # host-level Q-learning
    host_q: QLearningConfig = field(
        default_factory=lambda: QLearningConfig(alpha=0.6, gamma=0.9, epsilon=0.5)
    )

    # system-level Q-learning
    system_q: QLearningConfig = field(
        default_factory=lambda: QLearningConfig(alpha=0.4, gamma=0.9, epsilon=0.5)
    )

    # system-level reward parameters
    reward_b: float = 1.0

    # host-level reward shaping
    reward_all_correct: float = 2.0
    reward_all_wrong: float = -2.0
    reward_vote_correct: float = 0.5
    reward_vote_wrong: float = -0.5
    reward_majority_correct_bonus: float = 0.4
    reward_majority_wrong_penalty: float = -0.4

    # training / evaluation
    training_episodes: int = 250
    training_max_time: int = 6000
    n_mc_runs: int = 300
    max_time: int = 20000
    eval_greedy: bool = True

    seed: int = 42

    # calibration knobs for unspecified parts of article
    host_strategy_strength: float = 0.20
    audit_mode: str = "correct_both"   # punish_only | correct_bad_only | correct_both

    vote_energy_scale: float = 1.0
    audit_energy_scale: float = 1.0

    training_epsilon_decay: float = 0.997
    system_check_bias: float = 0.0

    use_majority_bonus: bool = True


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