from __future__ import annotations

from dataclasses import dataclass
from math import exp
import random
import statistics

from game_ext.qtcid_repro.mitchell.energy import (
    MitchellEnergyConfig,
    audit_energy,
    etids,
)
from game_ext.qtcid_repro.mitchell.voting import wang_ids_error_probability
from game_ext.qtcid_repro.types import SummaryStats
from game_ext.qtcid_repro.utils import binomial_sample, safe_div


@dataclass
class WangBVSConfig:
    n_nodes: int = 128
    n_neighbors: int = 32
    m_voters: int = 5
    tids: int = 200

    lambda_capture: float = 1 / 3600
    hpfn: float = 0.05
    hpfp: float = 0.05

    pa: float = 0.5
    pc: float = 0.5
    beta: float = 1.0

    et: float = 0.000125
    er: float = 0.00005
    ea: float = 0.00174
    es: float = 0.0005
    alpha_ranging: int = 5

    initial_system_energy: float = 16128000.0
    max_time: int = 20000
    runs: int = 200
    seed: int = 42


@dataclass
class WangBVSRunResult:
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
    audits_expected: float

    good_left: int
    bad_left: int
    evicted_left: int


class WangBVSSimulator:
    """
    Simplified Wang BVS baseline for Figure 3 (beta = 1).

    Important:
    - no life-layer runtime dynamics here
    - uses Wang Eq. (6)
    - uses audit effect only in the false-negative branch, as described by the SPN model
    - uses expected audit energy per IDS cycle: Pc * E_audit
    """

    def __init__(self, cfg: WangBVSConfig, seed: int) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)

        self.t = 0
        self.ng = cfg.n_nodes
        self.nb = 0
        self.ne = 0

        self.energy_left = cfg.initial_system_energy

        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.energy_spent_voting = 0.0
        self.energy_spent_audit = 0.0
        self.audits_expected = 0.0

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

        self.vote_energy_value = etids(self.energy_cfg)
        self.audit_energy_value = audit_energy(self.energy_cfg, contacted_fraction=0.5)

    def active_total(self) -> int:
        return self.ng + self.nb

    def byzantine_failure(self) -> bool:
        active = self.active_total()
        if active <= 0:
            return True
        return self.nb >= active / 3.0

    def energy_failure(self) -> bool:
        return self.energy_left <= 0.0

    def capture_step(self) -> None:
        if self.ng <= 0:
            return

        p_cap = 1.0 - exp(-self.cfg.lambda_capture * self.cfg.tids)
        new_bad = binomial_sample(self.rng, self.ng, p_cap)
        if new_bad > 0:
            self.ng -= new_bad
            self.nb += new_bad

    def ids_step(self) -> None:
        if self.active_total() < self.cfg.m_voters:
            return

        # Voting energy every IDS cycle
        self.energy_left -= self.vote_energy_value
        self.energy_spent_voting += self.vote_energy_value

        # Expected audit energy every IDS cycle
        expected_audit_energy = self.cfg.pc * self.audit_energy_value
        self.energy_left -= expected_audit_energy
        self.energy_spent_audit += expected_audit_energy
        self.audits_expected += self.cfg.pc

        p_fp_ids = wang_ids_error_probability(
            n_good=self.ng,
            n_bad=self.nb,
            m=self.cfg.m_voters,
            pa=self.cfg.pa,
            omega=self.cfg.hpfp,
        )

        p_fn_ids = wang_ids_error_probability(
            n_good=self.ng,
            n_bad=self.nb,
            m=self.cfg.m_voters,
            pa=self.cfg.pa,
            omega=self.cfg.hpfn,
        )

        # Good targets
        if self.ng > 0:
            fp_count = binomial_sample(self.rng, self.ng, p_fp_ids)
            tn_count = self.ng - fp_count

            self.ng -= fp_count
            self.ne += fp_count

            self.fp += fp_count
            self.tn += tn_count

        # Bad targets
        if self.nb > 0:
            if abs(self.cfg.beta - 1.0) > 1e-9:
                raise ValueError(
                    "This baseline Wang Figure 3 simulator supports only beta=1. "
                    "Layered beta=1/2 and beta=1/3 should be implemented separately."
                )

            # SPN logic for beta=1:
            # true positive with probability (1 - p_fn_ids)
            # false negative with probability p_fn_ids
            # among false negatives, audit catches the attacker with probability pa * pc
            # so bad node survives with probability p_fn_ids * (1 - pa * pc)
            p_bad_survive = p_fn_ids * (1.0 - self.cfg.pa * self.cfg.pc)
            p_bad_survive = max(0.0, min(1.0, p_bad_survive))

            bad_survive_count = binomial_sample(self.rng, self.nb, p_bad_survive)
            bad_removed_count = self.nb - bad_survive_count

            self.nb = bad_survive_count
            self.ne += bad_removed_count

            # bookkeeping
            # true positive = removed bad nodes
            # false negative = bad nodes still left in the system
            self.tp += bad_removed_count
            self.fn += bad_survive_count

    def run(self) -> WangBVSRunResult:
        while self.t < self.cfg.max_time:
            # Если система уже умерла в текущем состоянии — выходим
            if self.byzantine_failure() or self.energy_failure():
                break

            # Прошел следующий интервал TIDS
            self.t += self.cfg.tids

            # За интервал накапливаются новые компрометации
            self.capture_step()

            # КЛЮЧЕВОЙ МОМЕНТ:
            # если за время до следующего IDS bad-узлов стало слишком много,
            # система должна умереть ДО ids_step()
            if self.byzantine_failure() or self.energy_failure():
                break

            # Только если система еще жива — выполняем IDS voting
            self.ids_step()

            if self.byzantine_failure() or self.energy_failure():
                break

        return WangBVSRunResult(
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
            audits_expected=self.audits_expected,
            good_left=self.ng,
            bad_left=self.nb,
            evicted_left=self.ne,
        )


def summarize_wang(results: list[WangBVSRunResult], runs: int, initial_energy: float) -> SummaryStats:
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
        audits_mean=mean([r.audits_expected for r in results]),
        byzantine_fail_ratio=safe_div(sum(1 for r in results if r.byzantine_failed), len(results)),
        energy_fail_ratio=safe_div(sum(1 for r in results if r.energy_failed), len(results)),
        good_left_mean=mean([r.good_left for r in results]),
        bad_left_mean=mean([r.bad_left for r in results]),
        evicted_left_mean=mean([r.evicted_left for r in results]),
    )


def run_wang_bvs_monte_carlo(cfg: WangBVSConfig) -> SummaryStats:
    results: list[WangBVSRunResult] = []
    for i in range(cfg.runs):
        sim = WangBVSSimulator(cfg, seed=cfg.seed + i)
        results.append(sim.run())
    return summarize_wang(results, cfg.runs, cfg.initial_system_energy)