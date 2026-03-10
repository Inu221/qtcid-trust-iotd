from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import random

from .env import IoDEnvironment, NodeState


@dataclass
class VotingResult:
    target_id: int
    voters: List[int]
    votes_bad: int
    votes_good: int
    decision_is_bad: bool
    ground_truth_is_bad: bool


class BVSBaseline:
    def __init__(self, env: IoDEnvironment, m: int, seed: Optional[int] = None) -> None:
        self.env = env
        self.m = m
        self.rng = random.Random(seed)

    def _sample_voters(self, target_id: int) -> List[int]:
        neighbors = self.env.get_neighbors(target_id)
        if not neighbors:
            return []

        self.rng.shuffle(neighbors)
        selected = neighbors[: min(self.m, len(neighbors))]
        return [node.node_id for node in selected]

    def _single_vote(self, voter_id: int, target_id: int) -> bool:
        voter = self.env.nodes[voter_id]
        target = self.env.nodes[target_id]

        self.env.consume_energy(voter_id, self.env.tx_energy_cost)

        if target.state == NodeState.EVICTED:
            return True

        true_bad = target.state == NodeState.BAD

        if voter.state == NodeState.BAD:
            if self.rng.random() < self.env.malicious_vote_probability:
                return not true_bad

        return true_bad

    def inspect_target(self, target_id: int) -> Optional[VotingResult]:
        target = self.env.nodes[target_id]
        if target.state == NodeState.EVICTED or not target.alive:
            return None

        voters = self._sample_voters(target_id)
        if not voters:
            return None

        votes_bad = 0
        votes_good = 0

        for voter_id in voters:
            vote_bad = self._single_vote(voter_id, target_id)
            if vote_bad:
                votes_bad += 1
            else:
                votes_good += 1

        decision_is_bad = votes_bad > votes_good
        ground_truth_is_bad = target.state == NodeState.BAD

        if decision_is_bad:
            self.env.evict_node(target_id)

        return VotingResult(
            target_id=target_id,
            voters=voters,
            votes_bad=votes_bad,
            votes_good=votes_good,
            decision_is_bad=decision_is_bad,
            ground_truth_is_bad=ground_truth_is_bad,
        )

    def step(self) -> Dict[str, int]:
        active_targets = [node.node_id for node in self.env.active_nodes()]
        if not active_targets:
            return {
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "inspected": 0,
            }

        target_id = self.rng.choice(active_targets)
        result = self.inspect_target(target_id)

        stats = {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "inspected": 0,
        }

        if result is None:
            return stats

        stats["inspected"] = 1

        if result.decision_is_bad and result.ground_truth_is_bad:
            stats["tp"] = 1
        elif (not result.decision_is_bad) and (not result.ground_truth_is_bad):
            stats["tn"] = 1
        elif result.decision_is_bad and (not result.ground_truth_is_bad):
            stats["fp"] = 1
        elif (not result.decision_is_bad) and result.ground_truth_is_bad:
            stats["fn"] = 1

        return stats