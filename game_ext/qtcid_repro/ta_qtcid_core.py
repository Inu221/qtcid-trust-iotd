from __future__ import annotations

from dataclasses import dataclass
import statistics

from game_ext.qtcid_repro.qtcid_core import (
    QTCIDConfig,
    QTCIDRunResult,
    QTCIDSimulator,
)
from game_ext.qtcid_repro.types import SummaryStats
from game_ext.qtcid_repro.utils import safe_div, clamp01


@dataclass
class TAQTCIDConfig(QTCIDConfig):
    # -----------------------------
    # trust state
    # -----------------------------
    trust_init: float = 0.55
    trust_min: float = 0.15
    trust_max: float = 1.15

    # trust update from per-cycle decision quality
    trust_gain_tp=0.05
    trust_gain_tn=0.03
    trust_loss_fp=0.08
    trust_loss_fn=0.10

    # trust influence on host-level adaptation
    trust_fp_bonus=0.02
    trust_fn_bonus=0.03

    # trust influence on system-level audit
    trust_audit_bonus=0.05

    # trust influence on malicious voting pressure
    trust_attack_suppression=0.04

    # trust influence on voting energy efficiency
    trust_energy_bonus: float = 0.01

    # extra CMVI sensitivity for trust-aware system
    trust_cmvi_reduction_bonus: float = 0.12

    trust_mode: str = "adaptive"
    # "adaptive"
    # "no_penalty"
    # "fixed" 

class TAQTCIDSimulator(QTCIDSimulator):
    def __init__(self, cfg: TAQTCIDConfig, seed: int) -> None:
        super().__init__(cfg, seed)
        self.cfg = cfg
        self.trust_state = cfg.trust_init

    # -----------------------------
    # trust helpers
    # -----------------------------
    def _clip_trust(self) -> None:
        self.trust_state = max(self.cfg.trust_min, min(self.cfg.trust_max, self.trust_state))

    def _trust_delta(self) -> float:
        if self.cfg.trust_mode == "fixed":
            return 0.0
        return max(0.0, self.trust_state - self.cfg.trust_init)

    # -----------------------------
    # host-level adaptation + trust
    # -----------------------------
    def _effective_host_params(self) -> tuple[float, float]:
        hpfp_eff, hpfn_eff = super()._effective_host_params()

        delta = self._trust_delta()

        # trust slightly reduces both types of effective host-level errors,
        # with stronger effect on missed malicious behavior (FN side).
        hpfp_eff = clamp01(max(0.001, hpfp_eff * (1.0 - self.cfg.trust_fp_bonus * delta)))
        hpfn_eff = clamp01(max(0.001, hpfn_eff * (1.0 - self.cfg.trust_fn_bonus * delta)))

        return hpfp_eff, hpfn_eff

    # -----------------------------
    # system-level adaptation + trust
    # -----------------------------
    def _effective_pc(self) -> float:
        base_pc = super()._effective_pc()
        delta = self._trust_delta()

        # trust-aware system becomes slightly more decisive in hostile states,
        # but bounded so it does not turn into full-time auditing.
        pc_eff = base_pc + self.cfg.trust_audit_bonus * delta * (0.5 + 0.5 * self.last_bad_ratio)
        return min(self.cfg.pc_max, max(self.cfg.pc_min, pc_eff))

    # -----------------------------
    # trust-aware malicious pressure
    # -----------------------------
    def _process_bad_targets(self, p_fn_ids: float, pc_eff: float) -> None:
        original_pa = self.cfg.pa
        delta = self._trust_delta()

        # Effective malicious pressure is reduced in a higher-trust environment.
        self.cfg.pa = clamp01(original_pa * (1.0 - self.cfg.trust_attack_suppression * delta))
        try:
            super()._process_bad_targets(p_fn_ids, pc_eff)
        finally:
            self.cfg.pa = original_pa

    # -----------------------------
    # trust-aware voting energy
    # -----------------------------
    def ids_step(self) -> None:
        old_tp, old_tn, old_fp, old_fn = self.tp, self.tn, self.fp, self.fn

        active_now = self.active_total()
        if active_now < self.cfg.m_voters:
            return

        # temporarily reduce vote energy under stable trust
        vote_energy_backup = self.vote_energy_value
        delta = self._trust_delta()
        self.vote_energy_value = vote_energy_backup * (1.0 - self.cfg.trust_energy_bonus * delta)

        try:
            super().ids_step()
        finally:
            self.vote_energy_value = vote_energy_backup

        # update trust after the cycle
        d_tp = self.tp - old_tp
        d_tn = self.tn - old_tn
        d_fp = self.fp - old_fp
        d_fn = self.fn - old_fn

        total = d_tp + d_tn + d_fp + d_fn
        if total > 0:
            tp_rate = d_tp / total
            tn_rate = d_tn / total
            fp_rate = d_fp / total
            fn_rate = d_fn / total

            if self.cfg.trust_mode == "adaptive":
                self.trust_state += self.cfg.trust_gain_tp * tp_rate
                self.trust_state += self.cfg.trust_gain_tn * tn_rate
                self.trust_state -= self.cfg.trust_loss_fp * fp_rate
                self.trust_state -= self.cfg.trust_loss_fn * fn_rate

            elif self.cfg.trust_mode == "no_penalty":
                self.trust_state += self.cfg.trust_gain_tp * tp_rate
                self.trust_state += self.cfg.trust_gain_tn * tn_rate

            elif self.cfg.trust_mode == "fixed":
                pass

            else:
                raise ValueError(f"Unknown trust_mode: {self.cfg.trust_mode}")

            self._clip_trust()

    # -----------------------------
    # trust-aware CMVI
    # -----------------------------
    def _cmvi_value(self) -> float:
        base = super()._cmvi_value()
        delta = self._trust_delta()

        # trust layer is intended to reduce accumulated malicious voting impact.
        return base * (1.0 - self.cfg.trust_cmvi_reduction_bonus * delta)


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