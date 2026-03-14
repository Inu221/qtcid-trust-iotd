from __future__ import annotations

from dataclasses import dataclass
from math import exp
import random
import statistics

from game_ext.qtcid_repro.qtcid.deployment import random_deploy_nodes
from game_ext.qtcid_repro.qtcid.energy import (
    active_energy_budget_ratio,
    apply_energy_spent,
    audit_round_energy,
    total_energy_spent,
    voting_round_energy,
)
from game_ext.qtcid_repro.qtcid.host_level import HostLevelVotingAgent
from game_ext.qtcid_repro.qtcid.neighbors import active_neighbors_of, build_neighbor_map
from game_ext.qtcid_repro.qtcid.system_level import SystemLevelAuditAgent
from game_ext.qtcid_repro.types import Node, NodeState, QTCIDConfig, RunStats, SummaryStats
from game_ext.qtcid_repro.utils import bernoulli, random_choice_without_replacement, safe_div


class QTCIDSimulator:
    """
    Monte Carlo simulator following Q-TCID Algorithm 1 structure.

    Faithful elements already included:
    - random deployment in (L, W)
    - nodes fixed after deployment
    - GOOD/BAD/EVICTED states
    - capture process with Pcap rate
    - every TIDS: collaborative host-level + system-level IDS step
    - stop on Byzantine failure or too low energy

    Still operationalized (not fully explicit in paper):
    - exact host-level mapping from aV_t to voter behavior
    - exact reward shaping for partial correctness
    """

    def __init__(self, cfg: QTCIDConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.rng = random.Random(seed)

        self.nodes = random_deploy_nodes(cfg, self.rng)
        self.neighbor_map = build_neighbor_map(self.nodes, cfg.communication_range)

        self.host_agent = HostLevelVotingAgent(cfg)
        self.system_agent = SystemLevelAuditAgent(cfg)

        self.time = 0

        self.stats = RunStats(
            mttf=0.0,
            byzantine_failed=False,
            energy_failed=False,
        )

    def reset_states_for_run(self) -> None:
        for n in self.nodes:
            n.state = NodeState.GOOD
            n.energy = self.cfg.ein
            n.vote_history.clear()
            n.mismatch_count = 0
            n.participated_votes = 0
            n.targeted_count = 0

        self.time = 0
        self.stats = RunStats(
            mttf=0.0,
            byzantine_failed=False,
            energy_failed=False,
        )

    def active_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.is_active(self.cfg.emin)]

    def good_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.state == NodeState.GOOD and n.energy > self.cfg.emin]

    def bad_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.state == NodeState.BAD and n.energy > self.cfg.emin]

    def evicted_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.state == NodeState.EVICTED or n.energy <= self.cfg.emin]

    def byzantine_failure(self) -> bool:
        active = self.active_nodes()
        if not active:
            return True

        bad = [n for n in active if n.state == NodeState.BAD]
        return len(bad) >= len(active) / 3.0

    def energy_failure(self) -> bool:
        active = self.active_nodes()
        return len(active) == 0

    def capture_step(self) -> None:
        p_cap = 1.0 - exp(-self.cfg.pcap_rate * 1.0)

        for n in self.nodes:
            if n.state == NodeState.GOOD and n.energy > self.cfg.emin:
                if bernoulli(self.rng, p_cap):
                    n.state = NodeState.BAD

    def _record_target_outcome(self, target: Node, majority_label_good: bool) -> None:
        target.targeted_count += 1

        target_is_good = target.state == NodeState.GOOD
        target_is_bad = target.state == NodeState.BAD

        if target_is_good:
            if majority_label_good:
                self.stats.tn += 1
            else:
                self.stats.fp += 1
                target.state = NodeState.EVICTED
        elif target_is_bad:
            if majority_label_good:
                self.stats.fn += 1
            else:
                self.stats.tp += 1
                target.state = NodeState.EVICTED

    def _host_voting_round(self, target: Node, voters: list[Node]) -> tuple[bool, list[tuple[Node, str, str, bool, bool]]]:
        """
        Returns:
            majority_label_good,
            raw_decisions: [(voter, state, action, vote_for_good, explored), ...]
        """
        decisions: list[tuple[Node, str, str, bool, bool]] = []

        for voter in voters:
            voter.participated_votes += 1

            malicious_voter = voter.state == NodeState.BAD
            malicious_attack = malicious_voter and bernoulli(self.rng, self.cfg.pa)

            state, action, vote_for_good, explored = self.host_agent.decide_vote(
                voter=voter,
                target=target,
                malicious_voter=malicious_voter,
                malicious_attack=malicious_attack,
            )

            decisions.append((voter, state, action, vote_for_good, explored))

        votes_for_good = sum(1 for _, _, _, vfg, _ in decisions if vfg)
        votes_for_bad = len(decisions) - votes_for_good
        majority_label_good = votes_for_good >= votes_for_bad

        return majority_label_good, decisions

    def _update_host_learning(
        self,
        target: Node,
        majority_label_good: bool,
        decisions: list[tuple[Node, str, str, bool, bool]],
    ) -> None:
        target_is_good = target.state == NodeState.GOOD

        all_correct = True
        all_wrong = True

        for voter, _, _, vote_for_good, _ in decisions:
            correct = (target_is_good and vote_for_good) or ((not target_is_good) and (not vote_for_good))
            all_correct &= correct
            all_wrong &= (not correct)

        next_state_cache: dict[int, str] = {}

        for voter, state, action, vote_for_good, _ in decisions:
            reward = self.host_agent.reward_value(
                target_is_good=target_is_good,
                vote_for_good=vote_for_good,
                all_correct=all_correct,
                all_wrong=all_wrong,
            )

            voter.remember_vote(target.node_id, +1 if vote_for_good else -1)

            next_state = self.host_agent.build_state(voter, target)
            next_state_cache[voter.node_id] = next_state
            self.host_agent.update_after_round(state, action, reward, next_state)

    def _audit_round(
        self,
        target: Node,
        voters: list[Node],
        majority_label_good: bool,
        decisions: list[tuple[Node, str, str, bool, bool]],
    ) -> None:
        active = self.active_nodes()
        bad_ratio_before = safe_div(len(self.bad_nodes()), len(active))
        budget_ratio_before = active_energy_budget_ratio(self.nodes, self.cfg)

        attacked = False
        for voter, _, _, vote_for_good, _ in decisions:
            if voter.state == NodeState.BAD:
                target_is_bad = target.state == NodeState.BAD
                expected_honest_vote_for_good = (not target_is_bad)
                if vote_for_good != expected_honest_vote_for_good:
                    attacked = True
                    break

        decision = self.system_agent.step(
            bad_ratio=bad_ratio_before,
            audit_budget_ratio=budget_ratio_before,
            attacked=attacked,
            next_bad_ratio=bad_ratio_before,
            next_audit_budget_ratio=budget_ratio_before,
        )

        if decision.action == "check":
            self.stats.audits_total += 1
            spent = audit_round_energy(self.cfg, target, voters)
            self.stats.energy_spent_audit += spent
            self.stats.energy_spent_total += spent

            share = spent / max(1, len(voters) + 1)
            apply_energy_spent(target, share)
            for v in voters:
                apply_energy_spent(v, share)

            # If check + detected attack, punish mismatching malicious voters
            if attacked:
                target_is_bad = target.state == NodeState.BAD
                expected_honest_vote_for_good = (not target_is_bad)

                for voter, _, _, vote_for_good, _ in decisions:
                    if voter.state == NodeState.BAD and vote_for_good != expected_honest_vote_for_good:
                        voter.mismatch_count += 1
                        voter.state = NodeState.EVICTED

    def ids_cycle(self) -> None:
        active = self.active_nodes()
        if len(active) < self.cfg.m_voters + 1:
            return

        for target in list(active):
            if not target.is_active(self.cfg.emin):
                continue

            neighbor_ids = active_neighbors_of(
                node_id=target.node_id,
                nodes=self.nodes,
                neighbor_map=self.neighbor_map,
                emin=self.cfg.emin,
            )
            neighbor_ids = [nid for nid in neighbor_ids if nid != target.node_id]

            if len(neighbor_ids) < self.cfg.m_voters:
                continue

            chosen_ids = random_choice_without_replacement(self.rng, neighbor_ids, self.cfg.m_voters)
            voters = [self.nodes[nid] for nid in chosen_ids]

            # Voting energy
            e_vote = voting_round_energy(self.cfg, target, voters)
            self.stats.energy_spent_voting += e_vote
            self.stats.energy_spent_total += e_vote

            share = e_vote / max(1, len(voters) + 1)
            apply_energy_spent(target, share)
            for v in voters:
                apply_energy_spent(v, share)

            majority_label_good, decisions = self._host_voting_round(target, voters)
            self.stats.votes_total += 1

            # Host-level learning before state transition
            self._update_host_learning(target, majority_label_good, decisions)

            # System-level audit layer
            self._audit_round(target, voters, majority_label_good, decisions)

            # Final target outcome after host-level decision
            self._record_target_outcome(target, majority_label_good)

    def single_run(self) -> RunStats:
        self.reset_states_for_run()

        while self.time <= self.cfg.max_time:
            if self.byzantine_failure():
                self.stats.byzantine_failed = True
                break
            if self.energy_failure():
                self.stats.energy_failed = True
                break

            # Algorithm 1 lines 7-14: capture inspection
            self.capture_step()

            # Algorithm 1 line 15
            if self.time % self.cfg.tids == 0:
                self.ids_cycle()

                if self.byzantine_failure():
                    self.stats.byzantine_failed = True
                    break
                if self.energy_failure():
                    self.stats.energy_failed = True
                    break

            self.time += 1

        self.stats.mttf = float(self.time)
        self.stats.good_left = len(self.good_nodes())
        self.stats.bad_left = len(self.bad_nodes())
        self.stats.evicted_left = len(self.evicted_nodes())

        return self.stats


def summarize_runs(results: list[RunStats]) -> SummaryStats:
    def mean(values):
        return statistics.mean(values) if values else 0.0

    def std(values):
        return statistics.pstdev(values) if len(values) > 1 else 0.0

    acc_values = [r.accuracy() for r in results]
    precision_values = [r.precision() for r in results]
    recall_values = [r.recall() for r in results]
    fpr_values = [r.fpr() for r in results]
    fnr_values = [r.fnr() for r in results]

    return SummaryStats(
        runs=len(results),
        mttf_mean=mean([r.mttf for r in results]),
        mttf_std=std([r.mttf for r in results]),
        accuracy_mean=mean(acc_values),
        precision_mean=mean(precision_values),
        recall_mean=mean(recall_values),
        fpr_mean=mean(fpr_values),
        fnr_mean=mean(fnr_values),
        energy_spent_mean=mean([r.energy_spent_total for r in results]),
        energy_voting_mean=mean([r.energy_spent_voting for r in results]),
        energy_audit_mean=mean([r.energy_spent_audit for r in results]),
        audits_mean=mean([r.audits_total for r in results]),
        byzantine_fail_ratio=safe_div(sum(1 for r in results if r.byzantine_failed), len(results)),
        energy_fail_ratio=safe_div(sum(1 for r in results if r.energy_failed), len(results)),
        good_left_mean=mean([r.good_left for r in results]),
        bad_left_mean=mean([r.bad_left for r in results]),
        evicted_left_mean=mean([r.evicted_left for r in results]),
    )


def run_qtcid_monte_carlo(cfg: QTCIDConfig) -> SummaryStats:
    results: list[RunStats] = []

    for i in range(cfg.n_mc_runs):
        sim = QTCIDSimulator(cfg, seed=cfg.seed + i)
        results.append(sim.single_run())

    return summarize_runs(results)