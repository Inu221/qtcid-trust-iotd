from __future__ import annotations

from dataclasses import dataclass
import statistics

from game_ext.qtcid_repro.qtcid_core import (
    QTCIDConfig,
    QTCIDRunResult,
    QTCIDSimulator,
)
from game_ext.qtcid_repro.types import SummaryStats
from game_ext.qtcid_repro.utils import safe_div
from game_ext.qtcid_repro.wang.game import penalize_bad_node_from_layer


@dataclass
class TAQTCIDConfig(QTCIDConfig):
    w0: float = 1.0
    w_min: float = 0.05
    w_max: float = 2.0
    eta_plus: float = 0.05
    eta_minus: float = 0.30
    theta: float = 0.5


class TAQTCIDSimulator(QTCIDSimulator):
    def __init__(self, cfg: TAQTCIDConfig, seed: int) -> None:
        super().__init__(cfg, seed)
        self.cfg = cfg

        self.good_ids: set[int] = set(range(cfg.n_nodes))
        self.bad_layers: list[set[int]] = [set() for _ in self.layers]
        self.evicted_ids: set[int] = set()

        self.trust_weight: dict[int, float] = {
            node_id: cfg.w0 for node_id in range(cfg.n_nodes)
        }
        self.audit_update_count = 0
        self.collective_update_count = 0

    def _sync_counts(self) -> None:
        self.ng = len(self.good_ids)
        self.layers = [len(layer) for layer in self.bad_layers]
        self.ne = len(self.evicted_ids)

    def nb_total(self) -> int:
        return sum(len(layer) for layer in self.bad_layers)

    def _active_ids(self) -> list[int]:
        active = list(self.good_ids)
        for layer in self.bad_layers:
            active.extend(layer)
        return active

    def capture_step(self) -> None:
        from math import exp

        if not self.good_ids:
            self._sync_counts()
            return

        p_cap = 1.0 - exp(-self.cfg.lambda_capture * self.cfg.tids)
        captured = [
            node_id for node_id in list(self.good_ids)
            if self.rng.random() < p_cap
        ]

        for node_id in captured:
            self.good_ids.remove(node_id)
            self.bad_layers[0].add(node_id)

        self._sync_counts()

    def _cast_vote(
        self,
        voter_id: int,
        target_is_bad: bool,
        hpfp_eff: float,
        hpfn_eff: float,
    ) -> int:
        voter_is_bad = any(voter_id in layer for layer in self.bad_layers)
        if voter_is_bad and self.rng.random() < self.cfg.pa:
            return 0 if target_is_bad else 1

        if target_is_bad:
            return 0 if self.rng.random() < hpfn_eff else 1
        return 1 if self.rng.random() < hpfp_eff else 0

    def _weighted_vote(
        self,
        target_id: int,
        target_is_bad: bool,
        hpfp_eff: float,
        hpfn_eff: float,
    ) -> tuple[list[tuple[int, int]], float, int] | None:
        candidates = [node_id for node_id in self._active_ids() if node_id != target_id]
        if len(candidates) < self.cfg.m_voters:
            return None

        voters = self.rng.sample(candidates, self.cfg.m_voters)
        votes = [
            (voter_id, self._cast_vote(voter_id, target_is_bad, hpfp_eff, hpfn_eff))
            for voter_id in voters
        ]
        weight_sum = sum(self.trust_weight[voter_id] for voter_id, _ in votes)
        weighted_score = safe_div(
            sum(self.trust_weight[voter_id] * vote for voter_id, vote in votes),
            weight_sum,
        )
        decision = 1 if weighted_score >= self.cfg.theta else 0
        return votes, weighted_score, decision

    def _update_trust(
        self,
        votes: list[tuple[int, int]],
        confirmed_decision: int,
        confirmation_source: str,
    ) -> None:
        for voter_id, vote in votes:
            if vote == confirmed_decision:
                self.trust_weight[voter_id] = min(
                    self.cfg.w_max,
                    self.trust_weight[voter_id] + self.cfg.eta_plus,
                )
            else:
                self.trust_weight[voter_id] = max(
                    self.cfg.w_min,
                    self.trust_weight[voter_id] - self.cfg.eta_minus,
                )

        if confirmation_source == "audit":
            self.audit_update_count += 1
        elif confirmation_source == "collective_decision":
            self.collective_update_count += 1
        else:
            raise ValueError(f"Unknown confirmation_source: {confirmation_source}")

    def _evict_good(self, node_id: int) -> None:
        self.good_ids.remove(node_id)
        self.evicted_ids.add(node_id)
        self.fp += 1
        self.false_good_evictions += 1

    def _remove_bad_as_tp(self, layer_idx: int, node_id: int) -> None:
        self.bad_layers[layer_idx].remove(node_id)
        self.evicted_ids.add(node_id)
        self.tp += 1

    def _retain_bad(self, layer_idx: int, node_id: int) -> None:
        self.fn += 1
        self.bad_nodes_retained += 1

    def _penalize_bad_after_audit(self, layer_idx: int, node_id: int) -> None:
        self.bad_layers[layer_idx].remove(node_id)
        evicted, next_layer = penalize_bad_node_from_layer(layer_idx, self.cfg.beta)
        if evicted:
            self.evicted_ids.add(node_id)
            self.tp += 1
        else:
            self.bad_layers[next_layer].add(node_id)
            self._retain_bad(next_layer, node_id)

    def _process_good_targets_weighted(self, hpfp_eff: float, hpfn_eff: float) -> None:
        for target_id in list(self.good_ids):
            vote_result = self._weighted_vote(
                target_id=target_id,
                target_is_bad=False,
                hpfp_eff=hpfp_eff,
                hpfn_eff=hpfn_eff,
            )
            if vote_result is None or target_id not in self.good_ids:
                continue

            votes, _weighted_score, decision = vote_result
            self._update_trust(votes, decision, "collective_decision")

            if decision == 1:
                self._evict_good(target_id)
            else:
                self.tn += 1

    def _process_bad_targets_weighted(self, hpfp_eff: float, hpfn_eff: float, pc_eff: float) -> None:
        total_fn_candidates = 0
        total_attack_events = 0
        total_audited_attacks = 0

        for layer_idx in range(len(self.bad_layers)):
            for target_id in list(self.bad_layers[layer_idx]):
                vote_result = self._weighted_vote(
                    target_id=target_id,
                    target_is_bad=True,
                    hpfp_eff=hpfp_eff,
                    hpfn_eff=hpfn_eff,
                )
                if vote_result is None or target_id not in self.bad_layers[layer_idx]:
                    continue

                votes, _weighted_score, decision = vote_result
                if decision == 1:
                    self._update_trust(votes, decision, "collective_decision")
                    self._remove_bad_as_tp(layer_idx, target_id)
                    continue

                total_fn_candidates += 1
                attacked = self.rng.random() < self.cfg.pa
                if attacked:
                    total_attack_events += 1

                audited = self.rng.random() < pc_eff
                if audited:
                    self.audits += 1
                    self.energy_left -= self.audit_energy_value
                    self.energy_spent_audit += self.audit_energy_value

                if audited and attacked:
                    total_audited_attacks += 1
                    self.audit_mismatch_events += 1
                    self._update_trust(votes, 1, "audit")
                    self._penalize_bad_after_audit(layer_idx, target_id)
                else:
                    self._update_trust(votes, decision, "collective_decision")
                    self._retain_bad(layer_idx, target_id)

        self.last_attack_ratio = safe_div(total_attack_events, max(1, total_fn_candidates))
        self.last_audit_success_ratio = safe_div(total_audited_attacks, max(1, total_attack_events))

    def ids_step(self) -> None:
        active_now = self.active_total()
        if active_now < self.cfg.m_voters:
            return

        active_ratio = safe_div(active_now, max(1, self.cfg.n_nodes))
        interval_vote_energy = self.vote_energy_value * (0.35 + 0.65 * active_ratio)
        self.energy_left -= interval_vote_energy
        self.energy_spent_voting += interval_vote_energy

        hpfp_eff, hpfn_eff = self._effective_host_params()
        pc_eff = self._effective_pc()

        self._process_good_targets_weighted(hpfp_eff, hpfn_eff)
        self._process_bad_targets_weighted(hpfp_eff, hpfn_eff, pc_eff)
        self._sync_counts()
        self._update_state_metrics()


def summarize_taqtcid(results: list[QTCIDRunResult], runs: int, initial_energy: float) -> SummaryStats:
    def mean(values):
        return statistics.mean(values) if values else 0.0

    def std(values):
        return statistics.pstdev(values) if len(values) > 1 else 0.0

    tp_mean = mean([r.tp for r in results])
    tn_mean = mean([r.tn for r in results])
    fp_mean = mean([r.fp for r in results])
    fn_mean = mean([r.fn for r in results])

    total = tp_mean + tn_mean + fp_mean + fn_mean
    accuracy = safe_div(tp_mean + tn_mean, total)
    precision = safe_div(tp_mean, tp_mean + fp_mean)
    recall = safe_div(tp_mean, tp_mean + fn_mean)
    fpr = safe_div(fp_mean, fp_mean + tn_mean)
    fnr = safe_div(fn_mean, fn_mean + tp_mean)

    energy_left_mean = mean([r.energy_left for r in results])

    return SummaryStats(
        runs=runs,
        mttf_mean=mean([r.mttf for r in results]),
        mttf_std=std([r.mttf for r in results]),
        accuracy_mean=accuracy,
        precision_mean=precision,
        recall_mean=recall,
        fpr_mean=fpr,
        fnr_mean=fnr,
        energy_spent_mean=initial_energy - energy_left_mean,
        energy_voting_mean=mean([r.energy_spent_voting for r in results]),
        energy_audit_mean=mean([r.energy_spent_audit for r in results]),
        audits_mean=mean([r.audits for r in results]),
        byzantine_fail_ratio=safe_div(sum(1 for r in results if r.byzantine_failed), len(results)),
        energy_fail_ratio=safe_div(sum(1 for r in results if r.energy_failed), len(results)),
        good_left_mean=mean([r.good_left for r in results]),
        bad_left_mean=mean([r.bad_left for r in results]),
        evicted_left_mean=mean([r.evicted_left for r in results]),
    )


def run_taqtcid_monte_carlo(cfg: TAQTCIDConfig) -> SummaryStats:
    results: list[QTCIDRunResult] = []
    for i in range(cfg.runs):
        sim = TAQTCIDSimulator(cfg, seed=cfg.seed + i)
        results.append(sim.run())
    return summarize_taqtcid(results, cfg.runs, cfg.initial_system_energy)
