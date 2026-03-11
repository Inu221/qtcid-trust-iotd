from __future__ import annotations

from .config import WangCycleConfig


def eranging(cfg: WangCycleConfig) -> float:
    return cfg.alpha_ranging * (cfg.et + cfg.n_neighbors * (cfg.er + cfg.ea))


def esensing(cfg: WangCycleConfig) -> float:
    return cfg.n_neighbors * (cfg.es + cfg.ea)


def edetection(cfg: WangCycleConfig) -> float:
    return cfg.m * (cfg.et + cfg.n_neighbors * cfg.er) + cfg.m * (
        cfg.et + (cfg.m - 1) * (cfg.er + cfg.ea)
    )


def etids(cfg: WangCycleConfig) -> float:
    base = cfg.energy_scale * cfg.n_nodes * (
        eranging(cfg) + esensing(cfg) + edetection(cfg)
    )
    tids_penalty = 1.0 + 200.0 / max(cfg.tids, 1)
    return base * tids_penalty


def audit_energy(cfg: WangCycleConfig) -> float:
    return cfg.audit_cost_scale * 0.8 * etids(cfg)