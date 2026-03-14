from __future__ import annotations

from dataclasses import dataclass
import statistics

from game_ext.qtcid_repro.qtcid_enhanced_core import (
    QTCIDEnhancedConfig,
    QTCIDEnhancedRunResult,
    QTCIDEnhancedSimulator,
)
from game_ext.qtcid_repro.types import SummaryStats, NodeState
from game_ext.qtcid_repro.utils import safe_div, bernoulli

@dataclass
class QTCIDTrustConfig(QTCIDEnhancedConfig):
    trust_init: float = 0.6
    trust_min: float = 0.2
    trust_max: float = 1.2
    trust_gain: float = 0.04
    trust_loss: float = 0.08


class QTCIDTrustSimulator(QTCIDEnhancedSimulator):
    def __init__(self, cfg: QTCIDTrustConfig, seed: int) -> None:
        super().__init__(cfg, seed)
        self.cfg = cfg
        self.global_trust = cfg.trust_init

    def _clip_trust(self) -> None:
        self.global_trust = max(self.cfg.trust_min, min(self.cfg.trust_max, self.global_trust))

    def _effective_host_params(self):
        hpfp_eff, hpfn_eff, energy_scale, host_state, host_action = super()._effective_host_params()

        trust_bonus = self.global_trust - self.cfg.trust_init

        hpfp_eff = max(0.001, hpfp_eff * (1.0 - 0.10 * trust_bonus))
        hpfn_eff = max(0.001, hpfn_eff * (1.0 - 0.14 * trust_bonus))
        energy_scale = max(0.85, energy_scale * (1.0 - 0.04 * trust_bonus))

        return hpfp_eff, hpfn_eff, energy_scale, host_state, host_action

    def _process_bad_targets(self, p_fn_ids: float):
        audited_attacks, trusted_attacks = super()._process_bad_targets(p_fn_ids)

        if audited_attacks > 0:
            self.global_trust += self.cfg.trust_gain * audited_attacks
        if trusted_attacks > 0:
            self.global_trust -= self.cfg.trust_loss * trusted_attacks

        self._clip_trust()
        return audited_attacks, trusted_attacks


def summarize_qtcid_trust(results: list[QTCIDEnhancedRunResult], runs: int, initial_energy: float) -> SummaryStats:
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


def run_qtcid_trust_monte_carlo(cfg: QTCIDTrustConfig) -> SummaryStats:
    results: list[QTCIDEnhancedRunResult] = []

    for i in range(cfg.runs):
        sim = QTCIDTrustSimulator(cfg, seed=cfg.seed + i)
        results.append(sim.run())

    return summarize_qtcid_trust(results, cfg.runs, cfg.initial_system_energy)

def _host_voting_round(
    self,
    target,
    voters,
):
    decisions = []

    for voter in voters:
        voter.participated_votes += 1

        malicious_voter = voter.state == NodeState.BAD
        attacked = malicious_voter and bernoulli(self.rng, self.cfg.pa)

        state, action, vote_for_good, explored = self.host_agent.decide_vote(
            voter=voter,
            target=target,
            malicious_voter=malicious_voter,
            malicious_attack=attacked,
        )

        decisions.append((voter, state, action, vote_for_good, explored, attacked))

    # trust-weighted majority
    weight_good = 0.0
    weight_bad = 0.0

    for voter, _, _, vote_for_good, _, _ in decisions:
        w = self.global_trust if voter.state == NodeState.GOOD else max(0.2, 1.2 - self.global_trust)
        if vote_for_good:
            weight_good += w
        else:
            weight_bad += w

    majority_label_good = weight_good >= weight_bad
    return majority_label_good, decisions