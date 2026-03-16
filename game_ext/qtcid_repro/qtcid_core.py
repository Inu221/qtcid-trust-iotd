from __future__ import annotations

from dataclasses import dataclass
import random
import statistics

from game_ext.qtcid_repro.mitchell.energy import (
    MitchellEnergyConfig,
    audit_energy,
    etids,
)
from game_ext.qtcid_repro.mitchell.voting import wang_ids_error_probability
from game_ext.qtcid_repro.types import SummaryStats
from game_ext.qtcid_repro.utils import binomial_sample, safe_div, clamp01
from game_ext.qtcid_repro.wang.bvs_core import WangBVSConfig
from game_ext.qtcid_repro.wang.game import (
    num_life_layers,
    penalize_bad_node_from_layer,
    supported_beta,
)


@dataclass
class QTCIDConfig(WangBVSConfig):
    """
    Reproducible operationalization of Q-TCID ideas on top of Wang/BVS core.

    Key principles retained from Q-TCID article:
    - two-level adaptation:
        1) host-level adaptive voting quality
        2) system-level adaptive audit intensity
    - periodic IDS voting with TIDS
    - malicious voting with probability Pa
    - energy-aware behavior
    - Monte Carlo reliability estimation

    This is intentionally NOT a claim of exact author-code reconstruction.
    """

    # -----------------------------
    # host-level adaptation
    # -----------------------------
    host_fp_reduction_max: float = 0.05
    host_fn_reduction_max: float = 0.08
    host_bad_ratio_gain: float = 0.30
    host_attack_gain: float = 0.22
    host_fp_feedback_gain: float = 0.15
    host_fn_feedback_gain: float = 0.22

    # -----------------------------
    # system-level adaptation
    # -----------------------------
    pc_min: float = 0.05
    pc_max: float = 0.70
    audit_bad_ratio_gain: float = 0.30
    audit_attack_gain: float = 0.28
    audit_fn_gain: float = 0.20
    audit_energy_penalty: float = 0.42

    # -----------------------------
    # energy shaping
    # -----------------------------
    vote_energy_scale: float = 0.98
    audit_energy_scale: float = 0.98

    # -----------------------------
    # mild bonus over plain BVS
    # -----------------------------
    # Q-TCID should not become "magically perfect".
    # These terms only soften error rates and audit pressure.
    qtcid_tp_bonus: float = 0.01
    qtcid_fp_penalty_relief: float = 0.005

    # -----------------------------
    # CMVI weights
    # -----------------------------
    cmvi_fp_good_weight: float = 1.0
    cmvi_fn_bad_weight: float = 1.5
    cmvi_mismatch_weight: float = 0.5

    adaptation_tids_scale: float = 250.0

@dataclass
class QTCIDRunResult:
    mttf: float
    byzantine_failed: bool
    energy_failed: bool

    tp: int
    tn: int
    fp: int
    fn: int

    energy_left: float
    energy_spent_voting: float
    energy_spent_audit: float
    audits: int

    good_left: int
    bad_left: int
    evicted_left: int

    # extra security counters
    false_good_evictions: int
    bad_nodes_retained: int
    audit_mismatch_events: int
    cmvi: float


class QTCIDSimulator:
    def __init__(self, cfg: QTCIDConfig, seed: int) -> None:
        if not supported_beta(cfg.beta):
            raise ValueError("Supported beta values are 1, 1/2, 1/3")

        self.cfg = cfg
        self.rng = random.Random(seed)

        self.t = 0
        self.ng = cfg.n_nodes
        self.ne = 0

        # layer 0 = highest life quota
        self.layers = [0 for _ in range(num_life_layers(cfg.beta))]

        self.energy_left = cfg.initial_system_energy

        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.audits = 0
        self.energy_spent_voting = 0.0
        self.energy_spent_audit = 0.0

        # additional safety counters
        self.false_good_evictions = 0
        self.bad_nodes_retained = 0
        self.audit_mismatch_events = 0

        # dynamic state summaries
        self.last_bad_ratio = 0.0
        self.last_fp_ratio = 0.0
        self.last_fn_ratio = 0.0
        self.last_attack_ratio = 0.0
        self.last_audit_success_ratio = 0.0

        self.energy_cfg = MitchellEnergyConfig(
            n_nodes=cfg.n_nodes,
            n_neighbors=cfg.n_neighbors,
            m_voters=cfg.m_voters,
            et=cfg.et,
            er=cfg.er,
            ea=cfg.ea,
            es=cfg.es,
            alpha_ranging=cfg.alpha_ranging,
        )

        self.vote_energy_value = etids(self.energy_cfg) * cfg.vote_energy_scale
        self.audit_energy_value = audit_energy(
            self.energy_cfg,
            contacted_fraction=0.5,
        ) * cfg.audit_energy_scale

    # -----------------------------
    # basic state helpers
    # -----------------------------
    def _adaptation_decay(self) -> float:
        return self.cfg.adaptation_tids_scale / (self.cfg.adaptation_tids_scale + self.cfg.tids)

    def nb_total(self) -> int:
        return sum(self.layers)

    def active_total(self) -> int:
        return self.ng + self.nb_total()

    def byzantine_failure(self) -> bool:
        active = self.active_total()
        if active <= 0:
            return True
        return self.nb_total() >= active / 3.0

    def energy_failure(self) -> bool:
        return self.energy_left <= 0.0

    def _update_state_metrics(self) -> None:
        active = max(1, self.active_total())
        self.last_bad_ratio = safe_div(self.nb_total(), active)
        self.last_fp_ratio = safe_div(self.fp, max(1, self.fp + self.tn))
        self.last_fn_ratio = safe_div(self.fn, max(1, self.fn + self.tp))

    # -----------------------------
    # capture process
    # -----------------------------
    def capture_step(self) -> None:
        from math import exp

        if self.ng <= 0:
            return

        p_cap = 1.0 - exp(-self.cfg.lambda_capture * self.cfg.tids)
        new_bad = binomial_sample(self.rng, self.ng, p_cap)

        if new_bad > 0:
            self.ng -= new_bad
            self.layers[0] += new_bad

    # -----------------------------
    # host-level adaptation
    # -----------------------------
    def _effective_host_params(self) -> tuple[float, float]:
        """
        Host-level intelligent dynamic voting operationalization.

        We do not reconstruct the hidden Q-table from the paper.
        Instead we map host-level adaptation into dynamic effective
        host IDS error probabilities.
        """
        decay = self._adaptation_decay()

        fp_reduction = (
            self.cfg.host_fp_reduction_max
            * (
                self.cfg.host_bad_ratio_gain * self.last_bad_ratio
                + self.cfg.host_attack_gain * self.cfg.pa
                + self.cfg.host_fp_feedback_gain * max(0.0, 1.0 - self.last_fp_ratio)
            )
        ) * decay

        fn_reduction = (
            self.cfg.host_fn_reduction_max
            * (
                self.cfg.host_bad_ratio_gain * self.last_bad_ratio
                + self.cfg.host_attack_gain * self.cfg.pa
                + self.cfg.host_fn_feedback_gain * max(0.0, 1.0 - self.last_fn_ratio)
            )
        ) * decay

        # keep reductions moderate
        fp_reduction = min(0.35, fp_reduction)
        fn_reduction = min(0.45, fn_reduction)

        hpfp_eff = clamp01(max(0.001, self.cfg.hpfp * (1.0 - fp_reduction)))
        hpfn_eff = clamp01(max(0.001, self.cfg.hpfn * (1.0 - fn_reduction)))

        return hpfp_eff, hpfn_eff

    # -----------------------------
    # system-level adaptation
    # -----------------------------
    def _effective_pc(self) -> float:
        """
        System-level intelligent audit operationalization.

        pc_eff rises when:
        - bad ratio is high
        - attack probability Pa is high
        - false negatives accumulate

        pc_eff is reduced when energy is getting depleted.
        """
        self._update_state_metrics()

        energy_ratio = safe_div(self.energy_left, self.cfg.initial_system_energy)

        decay = self._adaptation_decay()

        score = (
            self.cfg.pc
            + decay * (
                self.cfg.audit_bad_ratio_gain * self.last_bad_ratio
                + self.cfg.audit_attack_gain * self.cfg.pa
                + self.cfg.audit_fn_gain * self.last_fn_ratio
            )
            - self.cfg.audit_energy_penalty * (1.0 - energy_ratio)
        )

        return min(self.cfg.pc_max, max(self.cfg.pc_min, score))

    # -----------------------------
    # system voting error probabilities
    # -----------------------------
    def _system_error_probabilities(self) -> tuple[float, float]:
        nb = self.nb_total()
        hpfp_eff, hpfn_eff = self._effective_host_params()

        p_fp_ids = wang_ids_error_probability(
            n_good=self.ng,
            n_bad=nb,
            m=self.cfg.m_voters,
            pa=self.cfg.pa,
            omega=hpfp_eff,
        )

        p_fn_ids = wang_ids_error_probability(
            n_good=self.ng,
            n_bad=nb,
            m=self.cfg.m_voters,
            pa=self.cfg.pa,
            omega=hpfn_eff,
        )

        # mild extra Q-TCID benefit, but bounded and realistic
        decay = self._adaptation_decay()

        fp_relief = self.cfg.qtcid_fp_penalty_relief * decay
        tp_bonus = self.cfg.qtcid_tp_bonus * decay

        p_fp_ids = clamp01(max(0.0, p_fp_ids * (1.0 - fp_relief)))
        p_fn_ids = clamp01(max(0.0, p_fn_ids * (1.0 - tp_bonus)))

        return p_fp_ids, p_fn_ids

    # -----------------------------
    # good target processing
    # -----------------------------
    def _process_good_targets(self, p_fp_ids: float) -> None:
        if self.ng <= 0:
            return

        fp_count = binomial_sample(self.rng, self.ng, p_fp_ids)
        tn_count = self.ng - fp_count

        self.ng -= fp_count
        self.ne += fp_count

        self.fp += fp_count
        self.tn += tn_count

        self.false_good_evictions += fp_count

    # -----------------------------
    # bad target processing
    # -----------------------------
    def _process_bad_targets(self, p_fn_ids: float, pc_eff: float) -> None:
        if self.nb_total() <= 0:
            self.last_attack_ratio = 0.0
            self.last_audit_success_ratio = 0.0
            return

        total_fn_candidates = 0
        total_attack_events = 0
        total_audited_attacks = 0

        new_layers = [0 for _ in self.layers]

        for layer_idx, count in enumerate(self.layers):
            if count <= 0:
                continue

            tp_count = binomial_sample(self.rng, count, 1.0 - p_fn_ids)
            fn_count = count - tp_count

            self.tp += tp_count
            self.ne += tp_count

            if fn_count <= 0:
                continue

            total_fn_candidates += fn_count

            stay_count = 0
            punished_count = 0

            for _ in range(fn_count):
                attacked = self.rng.random() < self.cfg.pa
                if attacked:
                    total_attack_events += 1

                audited = self.rng.random() < pc_eff
                if audited:
                    self.audits += 1
                    self.energy_left -= self.audit_energy_value
                    self.energy_spent_audit += self.audit_energy_value

                if audited and attacked:
                    punished_count += 1
                    total_audited_attacks += 1
                    self.audit_mismatch_events += 1
                else:
                    stay_count += 1

            for _ in range(punished_count):
                evicted, next_layer = penalize_bad_node_from_layer(layer_idx, self.cfg.beta)
                if evicted:
                    self.ne += 1
                    self.tp += 1
                else:
                    new_layers[next_layer] += 1
                    self.fn += 1
                    self.bad_nodes_retained += 1

            if stay_count > 0:
                new_layers[layer_idx] += stay_count
                self.fn += stay_count
                self.bad_nodes_retained += stay_count

        self.layers = new_layers

        self.last_attack_ratio = safe_div(total_attack_events, max(1, total_fn_candidates))
        self.last_audit_success_ratio = safe_div(total_audited_attacks, max(1, total_attack_events))

    # -----------------------------
    # one IDS step
    # -----------------------------
    def ids_step(self) -> None:
        active_now = self.active_total()
        if active_now < self.cfg.m_voters:
            return

        # same high-level Wang logic: one IDS cycle covers the current system
        active_ratio = safe_div(active_now, max(1, self.cfg.n_nodes))
        interval_vote_energy = self.vote_energy_value * (0.35 + 0.65 * active_ratio)
        self.energy_left -= interval_vote_energy
        self.energy_spent_voting += interval_vote_energy

        p_fp_ids, p_fn_ids = self._system_error_probabilities()
        pc_eff = self._effective_pc()

        self._process_good_targets(p_fp_ids)
        self._process_bad_targets(p_fn_ids, pc_eff)
        self._update_state_metrics()

    # -----------------------------
    # metric
    # -----------------------------
    def _cmvi_value(self) -> float:
        return (
            self.cfg.cmvi_fp_good_weight * self.false_good_evictions
            + self.cfg.cmvi_fn_bad_weight * self.bad_nodes_retained
            + self.cfg.cmvi_mismatch_weight * self.audit_mismatch_events
        )

    # -----------------------------
    # run loop
    # -----------------------------
    def run(self) -> QTCIDRunResult:
        while self.t <= self.cfg.max_time:
            if self.byzantine_failure() or self.energy_failure():
                break

            self.capture_step()
            self.ids_step()

            if self.byzantine_failure() or self.energy_failure():
                break

            self.t += self.cfg.tids

        return QTCIDRunResult(
            mttf=float(self.t),
            byzantine_failed=self.byzantine_failure(),
            energy_failed=self.energy_failure(),
            tp=self.tp,
            tn=self.tn,
            fp=self.fp,
            fn=self.fn,
            energy_left=self.energy_left,
            energy_spent_voting=self.energy_spent_voting,
            energy_spent_audit=self.energy_spent_audit,
            audits=self.audits,
            good_left=self.ng,
            bad_left=self.nb_total(),
            evicted_left=self.ne,
            false_good_evictions=self.false_good_evictions,
            bad_nodes_retained=self.bad_nodes_retained,
            audit_mismatch_events=self.audit_mismatch_events,
            cmvi=self._cmvi_value(),
        )


def summarize_qtcid(results: list[QTCIDRunResult], runs: int, initial_energy: float) -> SummaryStats:
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


def run_qtcid_monte_carlo(cfg: QTCIDConfig) -> SummaryStats:
    results: list[QTCIDRunResult] = []
    for i in range(cfg.runs):
        sim = QTCIDSimulator(cfg, seed=cfg.seed + i)
        results.append(sim.run())
    return summarize_qtcid(results, cfg.runs, cfg.initial_system_energy)