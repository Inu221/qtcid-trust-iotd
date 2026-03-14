from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MitchellEnergyConfig:
    n_nodes: int
    n_neighbors: int
    m_voters: int
    et: float
    er: float
    ea: float
    es: float
    alpha_ranging: int


def eranging(cfg: MitchellEnergyConfig) -> float:
    return cfg.alpha_ranging * (cfg.et + cfg.n_neighbors * (cfg.er + cfg.ea))


def esensing(cfg: MitchellEnergyConfig) -> float:
    return cfg.n_neighbors * (cfg.es + cfg.ea)


def edetection(cfg: MitchellEnergyConfig) -> float:
    return cfg.m_voters * (cfg.et + cfg.n_neighbors * cfg.er) + cfg.m_voters * (
        cfg.et + (cfg.m_voters - 1) * (cfg.er + cfg.ea)
    )


def etids(cfg: MitchellEnergyConfig) -> float:
    return cfg.n_nodes * (eranging(cfg) + esensing(cfg) + edetection(cfg))


def audit_energy(cfg: MitchellEnergyConfig, contacted_fraction: float = 0.5) -> float:
    return contacted_fraction * etids(cfg)