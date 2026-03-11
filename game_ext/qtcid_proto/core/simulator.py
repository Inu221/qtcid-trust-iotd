from __future__ import annotations

from dataclasses import dataclass
from math import exp
import random

from .config import WangCycleConfig
from .energy import etids, audit_energy
from .utils import binomial_sample, num_layers, safe_div
from .voting import wang_ids_error_probability
from .policies import BasePolicy, SOIDPolicy, QTCIDPolicy
from .metrics import summarize


@dataclass
class RunResult:
    mttf: float
    energy_left: float
    byzantine_failed: bool
    energy_failed: bool
    good_left: int
    bad_left: int
    evicted_left: int
    tp: int
    tn: int
    fp: int
    fn: int
    audits: int
    vote_energy: float
    audit_energy_total: float


class WangCycleSimulator:
    def __init__(self, cfg: WangCycleConfig, seed: int, policy) -> None:
        self.cfg = cfg
        self.policy = policy
        self.rng = random.Random(seed)

        self.t = 0
        self.ng = cfg.n_nodes
        self.ne = 0
        self.layers = [0 for _ in range(num_layers(cfg.beta))]
        self.energy_left = cfg.initial_system_energy

        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.audits = 0
        self.vote_energy = 0.0
        self.audit_energy_total = 0.0

        self.vote_energy_value = etids(cfg)
        self.audit_energy_value = audit_energy(cfg)

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

    def binomial(self, n: int, p: float) -> int:
        return binomial_sample(self.rng, n, p)

    def audit_state(self, layer_idx: int) -> str:
        bad_ratio = safe_div(self.nb_total(), max(1, self.active_total()))
        pressure = "low" if bad_ratio < 0.10 else "mid" if bad_ratio < 0.20 else "high"
        return f"L{layer_idx}|{pressure}"

    def capture_step(self) -> None:
        if self.ng <= 0:
            return
        p_cap = 1.0 - exp(-self.cfg.lambda_capture * self.cfg.tids)
        newly_bad = self.binomial(self.ng, p_cap)
        if newly_bad > 0:
            self.ng -= newly_bad
            self.layers[0] += newly_bad

    def ids_step(self) -> None:
        if self.active_total() < self.cfg.m:
            return

        tp0, tn0, fp0, fn0 = self.tp, self.tn, self.fp, self.fn
        e0 = self.energy_left

        hpfp_eff, hpfn_eff, host_energy_scale = self.policy.host_params(self)
        vote_spent = host_energy_scale * self.vote_energy_value
        self.energy_left -= vote_spent
        self.vote_energy += vote_spent

        n_bad = self.nb_total()
        p_fp = wang_ids_error_probability(self.ng, n_bad, self.cfg.m, self.cfg.pa, hpfp_eff)
        p_fn = wang_ids_error_probability(self.ng, n_bad, self.cfg.m, self.cfg.pa, hpfn_eff)

        good_before = self.ng
        fp_count = self.binomial(good_before, p_fp)
        tn_count = good_before - fp_count

        self.ng -= fp_count
        self.ne += fp_count
        self.fp += fp_count
        self.tn += tn_count

        new_layers = [0 for _ in self.layers]

        for layer_idx, count in enumerate(self.layers):
            if count <= 0:
                continue

            tp_count = self.binomial(count, 1.0 - p_fn)
            remaining_fn = count - tp_count

            self.tp += tp_count
            self.ne += tp_count
            self.policy.handle_placeholder(self, layer_idx, remaining_fn, new_layers)

        self.layers = new_layers

        if hasattr(self.policy, "finalize_host_learning"):
            self.policy.finalize_host_learning(
                self,
                self.tp - tp0,
                self.tn - tn0,
                self.fp - fp0,
                self.fn - fn0,
                e0 - self.energy_left,
            )

    def run(self, stop_on_byzantine: bool = True, horizon: int | None = None) -> RunResult:
        limit = self.cfg.max_time if horizon is None else horizon

        while self.t <= limit:
            if self.energy_failure() or (stop_on_byzantine and self.byzantine_failure()):
                break

            self.capture_step()
            self.ids_step()

            if self.energy_failure() or (stop_on_byzantine and self.byzantine_failure()):
                break

            self.t += self.cfg.tids

        return RunResult(
            mttf=float(self.t),
            energy_left=self.energy_left,
            byzantine_failed=self.byzantine_failure(),
            energy_failed=self.energy_failure(),
            good_left=self.ng,
            bad_left=self.nb_total(),
            evicted_left=self.ne,
            tp=self.tp,
            tn=self.tn,
            fp=self.fp,
            fn=self.fn,
            audits=self.audits,
            vote_energy=self.vote_energy,
            audit_energy_total=self.audit_energy_total,
        )


def _run_many(cfg: WangCycleConfig, policy_factory, stop_on_byzantine: bool = True, horizon: int | None = None) -> dict:
    results = []
    base_seed = cfg.seed or 0
    for i in range(cfg.runs):
        policy = policy_factory(cfg, base_seed + i)
        sim = WangCycleSimulator(cfg, base_seed + i, policy)
        results.append(sim.run(stop_on_byzantine=stop_on_byzantine, horizon=horizon))
    return summarize(results, cfg.runs, cfg.initial_system_energy)


def run_bvs(cfg: WangCycleConfig) -> dict:
    return _run_many(cfg, lambda cfg, seed: BasePolicy())


def run_soid(cfg: WangCycleConfig) -> dict:
    return _run_many(cfg, lambda cfg, seed: SOIDPolicy())


def run_qtcid(cfg: WangCycleConfig) -> dict:
    return _run_many(cfg, lambda cfg, seed: QTCIDPolicy(cfg, seed))


def run_bvs_energy(cfg: WangCycleConfig) -> dict:
    return _run_many(cfg, lambda cfg, seed: BasePolicy(), stop_on_byzantine=False, horizon=cfg.energy_eval_horizon)


def run_soid_energy(cfg: WangCycleConfig) -> dict:
    return _run_many(cfg, lambda cfg, seed: SOIDPolicy(), stop_on_byzantine=False, horizon=cfg.energy_eval_horizon)


def run_qtcid_energy(cfg: WangCycleConfig) -> dict:
    return _run_many(cfg, lambda cfg, seed: QTCIDPolicy(cfg, seed), stop_on_byzantine=False, horizon=cfg.energy_eval_horizon)