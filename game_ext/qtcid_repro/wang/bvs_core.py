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
from game_ext.qtcid_repro.wang.game import (
    num_life_layers,
    penalize_bad_node_from_layer,
    pmin_c,
    supported_beta,
    theorem1_attack_discouraged,
)


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

    def validate(self) -> None:
        if not supported_beta(self.beta):
            raise ValueError("Supported beta values are 1, 1/2, 1/3")
        if self.m_voters <= 0:
            raise ValueError("m_voters must be positive")


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
    audits: int

    good_left: int
    bad_left: int
    evicted_left: int


class WangBVSSimulator:
    """
    Faithful Wang-style BVS / IDS attack-defense game core.

    State:
      - Ng: number of good nodes
      - Ne: number of evicted nodes
      - layers[k]: number of bad nodes in life-quota layer k

    For beta = 1:
      layers = [Nb^1]
    For beta = 1/2:
      layers = [Nb^1, Nb^2]
    For beta = 1/3:
      layers = [Nb^1, Nb^2, Nb^3]
    """

    def __init__(self, cfg: WangBVSConfig, seed: int) -> None:
        cfg.validate()
        self.cfg = cfg
        self.rng = random.Random(seed)

        self.t = 0
        self.ng = cfg.n_nodes
        self.ne = 0
        self.layers = [0 for _ in range(num_life_layers(cfg.beta))]

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
        # Wang says audit contacts approximately half of all nodes
        self.audit_energy_value = audit_energy(self.energy_cfg, contacted_fraction=0.5)

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

    def capture_step(self) -> None:
        if self.ng <= 0:
            return

        p_cap = 1.0 - exp(-self.cfg.lambda_capture * self.cfg.tids)
        new_bad = binomial_sample(self.rng, self.ng, p_cap)

        if new_bad > 0:
            self.ng -= new_bad
            # all newly compromised nodes enter the top bad layer
            self.layers[0] += new_bad

    def _system_error_probabilities(self) -> tuple[float, float]:
        nb = self.nb_total()

        p_fp_ids = wang_ids_error_probability(
            n_good=self.ng,
            n_bad=nb,
            m=self.cfg.m_voters,
            pa=self.cfg.pa,
            omega=self.cfg.hpfp,
        )

        p_fn_ids = wang_ids_error_probability(
            n_good=self.ng,
            n_bad=nb,
            m=self.cfg.m_voters,
            pa=self.cfg.pa,
            omega=self.cfg.hpfn,
        )

        return p_fp_ids, p_fn_ids

    def _process_good_targets(self, p_fp_ids: float) -> None:
        if self.ng <= 0:
            return

        fp_count = binomial_sample(self.rng, self.ng, p_fp_ids)
        tn_count = self.ng - fp_count

        self.ng -= fp_count
        self.ne += fp_count

        self.fp += fp_count
        self.tn += tn_count

    def _process_bad_targets(self, p_fn_ids: float) -> None:
        if self.nb_total() <= 0:
            return

        new_layers = [0 for _ in self.layers]

        for layer_idx, count in enumerate(self.layers):
            if count <= 0:
                continue

            # True positives: bad node correctly evicted by voting
            tp_count = binomial_sample(self.rng, count, 1.0 - p_fn_ids)
            fn_count = count - tp_count

            self.tp += tp_count
            self.ne += tp_count

            if fn_count <= 0:
                continue

            # False-negative bad nodes enter placeholder and are then routed by Wang game logic
            audited_attack_count = 0
            stay_count = 0

            for _ in range(fn_count):
                audited = self.rng.random() < self.cfg.pc
                attacked = self.rng.random() < self.cfg.pa

                if audited:
                    self.audits += 1
                    self.energy_left -= self.audit_energy_value
                    self.energy_spent_audit += self.audit_energy_value

                # Wang placeholder routing:
                # with probability Pa*Pc the bad node is penalized;
                # otherwise it remains in system
                if audited and attacked:
                    audited_attack_count += 1
                else:
                    stay_count += 1

            # Penalized attackers move down a life-quota layer or get evicted
            for _ in range(audited_attack_count):
                evicted, next_layer = penalize_bad_node_from_layer(layer_idx, self.cfg.beta)
                if evicted:
                    self.ne += 1
                    self.tp += 1
                else:
                    new_layers[next_layer] += 1
                    self.fn += 1

            # Non-penalized false negatives remain in same layer
            if stay_count > 0:
                new_layers[layer_idx] += stay_count
                self.fn += stay_count

        self.layers = new_layers

    def ids_step(self) -> None:
        if self.active_total() < self.cfg.m_voters:
            return

        self.energy_left -= self.vote_energy_value
        self.energy_spent_voting += self.vote_energy_value

        p_fp_ids, p_fn_ids = self._system_error_probabilities()
        self._process_good_targets(p_fp_ids)
        self._process_bad_targets(p_fn_ids)

    def run(self) -> WangBVSRunResult:
        while self.t <= self.cfg.max_time:
            if self.byzantine_failure() or self.energy_failure():
                break

            self.capture_step()
            self.ids_step()

            if self.byzantine_failure() or self.energy_failure():
                break

            self.t += self.cfg.tids

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
            audits=self.audits,
            good_left=self.ng,
            bad_left=self.nb_total(),
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
        audits_mean=mean([r.audits for r in results]),
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