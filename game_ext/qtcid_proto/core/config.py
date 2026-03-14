from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class WangCycleConfig:
    n_nodes: int = 128
    n_neighbors: int = 32
    m: int = 5
    tids: int = 200
    lambda_capture: float = 1 / 3600
    pa: float = 0.5
    pc: float = 0.5
    beta: float = 1.0
    hpfn: float = 0.05
    hpfp: float = 0.05
    alpha_ranging: int = 5
    et: float = 0.000125
    er: float = 0.00005
    ea: float = 0.00174
    es: float = 0.0005
    initial_system_energy: float = 16128000.0
    energy_scale: float = 1.0
    alpha_s: float = 0.4
    gamma_s: float = 0.9
    epsilon_s: float = 0.5
    alpha_n: float = 0.6
    gamma_n: float = 0.9
    epsilon_n: float = 0.5
    reward_b: float = 1.0
    max_time: int = 14000
    energy_eval_horizon: int = 4000
    runs: int = 200
    seed: Optional[int] = 42
    audit_cost_scale: float = 1.0

    trust_init: float = 0.6
    trust_min: float = 0.2
    trust_max: float = 1.2
    trust_gain: float = 0.04
    trust_loss: float = 0.08

def make_bvs_config(pa: float, tids: int, runs: int = 200, seed: int = 42) -> WangCycleConfig:
    return WangCycleConfig(
        pa=pa,
        tids=tids,
        pc=0.0,
        beta=1.0,
        runs=runs,
        seed=seed,
    )


def make_soid_config(pa: float, tids: int, runs: int = 200, seed: int = 42) -> WangCycleConfig:
    return WangCycleConfig(
        pa=pa,
        tids=tids,
        pc=0.5,
        beta=1.0,
        runs=runs,
        seed=seed,
    )


def make_qtcid_config(pa: float, tids: int, runs: int = 200, seed: int = 42) -> WangCycleConfig:
    return WangCycleConfig(
        pa=pa,
        tids=tids,
        pc=0.5,
        beta=1.0,
        runs=runs,
        seed=seed,
        reward_b=1.2,
    )

def make_qtcid_trust_config(pa: float, tids: int, runs: int = 200, seed: int = 42) -> WangCycleConfig:
    return WangCycleConfig(
        pa=pa,
        tids=tids,
        pc=0.5,
        beta=1.0,
        runs=runs,
        seed=seed,
        reward_b=1.25,
        trust_init=0.6,
        trust_min=0.2,
        trust_max=1.2,
        trust_gain=0.04,
        trust_loss=0.08,
    )