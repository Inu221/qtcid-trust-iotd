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
from game_ext.qtcid_repro.mitchell.voting import voting_error_probability_basic
from game_ext.qtcid_repro.types import SummaryStats
from game_ext.qtcid_repro.utils import binomial_sample, safe_div


@dataclass
class MitchellConfig:
    n_nodes: int = 128
    n_neighbors: int = 32
    m_voters: int = 5
    tids: int = 200

    lambda_capture: float = 1 / 3600
    hpfn: float = 0.05
    hpfp: float = 0.05

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
class MitchellRunResult:
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


class MitchellSimulator:
    """
    Baseline distributed IDS engine.
    survivability/voting skeleton:
    - good nodes may be captured
    - IDS voting every TIDS
    - good nodes can be falsely evicted
    - bad nodes can be correctly evicted or remain
    - Byzantine failure / energy depletion stop the run
    """

    def __init__(self, cfg: MitchellConfig, seed: int) -> None:
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

        self.audits = 0
        self.energy_spent_voting = 0.0
        self.energy_spent_audit = 0.0

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

        self.energy_left -= self.vote_energy_value
        self.energy_spent_voting += self.vote_energy_value

        # Target GOOD -> false positive probability at system level
        p_fp_ids = voting_error_probability_basic(
            n_good=self.ng,
            n_bad=self.nb,
            m=self.cfg.m_voters,
            omega=self.cfg.hpfp,
        )

        # Target BAD -> false negative probability at system level
        p_fn_ids = voting_error_probability_basic(
            n_good=self.ng,
            n_bad=self.nb,
            m=self.cfg.m_voters,
            omega=self.cfg.hpfn,
        )

        # Good nodes evaluated by IDS
        if self.ng > 0:
            fp_count = binomial_sample(self.rng, self.ng, p_fp_ids)
            tn_count = self.ng - fp_count

            self.ng -= fp_count
            self.ne += fp_count

            self.fp += fp_count
            self.tn += tn_count

        # Bad nodes evaluated by IDS
        if self.nb > 0:
            tp_count = binomial_sample(self.rng, self.nb, 1.0 - p_fn_ids)
            fn_count = self.nb - tp_count

            self.nb = fn_count
            self.ne += tp_count

            self.tp += tp_count
            self.fn += fn_count

    def run(self) -> MitchellRunResult:
        while self.t <= self.cfg.max_time:
            if self.byzantine_failure() or self.energy_failure():
                break

            self.capture_step()
            self.ids_step()

            if self.byzantine_failure() or self.energy_failure():
                break

            self.t += self.cfg.tids

        return MitchellRunResult(
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
            bad_left=self.nb,
            evicted_left=self.ne,
        )


def summarize_mitchell(results: list[MitchellRunResult], runs: int, initial_energy: float) -> SummaryStats:
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


def run_mitchell_monte_carlo(cfg: MitchellConfig) -> SummaryStats:
    results: list[MitchellRunResult] = []

    for i in range(cfg.runs):
        sim = MitchellSimulator(cfg, seed=cfg.seed + i)
        results.append(sim.run())

    return summarize_mitchell(results, cfg.runs, cfg.initial_system_energy)