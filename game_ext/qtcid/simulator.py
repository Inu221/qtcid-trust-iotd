from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import statistics

from .env import IoDEnvironment
from .baselines import BVSBaseline
from .metrics import compute_classification_metrics


@dataclass
class SimulationConfig:
    n_nodes: int = 30
    area_size: float = 100.0
    comm_radius: float = 35.0
    initial_energy: float = 100.0
    capture_probability: float = 0.02
    malicious_vote_probability: float = 0.2
    tx_energy_cost: float = 0.5
    audit_energy_cost: float = 1.0
    m: int = 5
    max_steps: int = 500
    runs: int = 30
    seed: Optional[int] = 42


def run_single_bvs_simulation(cfg: SimulationConfig, run_seed: int) -> Dict[str, float]:
    env = IoDEnvironment(
        n_nodes=cfg.n_nodes,
        area_size=cfg.area_size,
        comm_radius=cfg.comm_radius,
        initial_energy=cfg.initial_energy,
        capture_probability=cfg.capture_probability,
        malicious_vote_probability=cfg.malicious_vote_probability,
        tx_energy_cost=cfg.tx_energy_cost,
        audit_energy_cost=cfg.audit_energy_cost,
        seed=run_seed,
    )

    bvs = BVSBaseline(env=env, m=cfg.m, seed=run_seed + 1000)

    tp = tn = fp = fn = 0
    total_energy_spent = 0.0
    steps_survived = 0

    for step in range(cfg.max_steps):
        env.maybe_capture_nodes()

        before_energy = sum(node.energy for node in env.nodes)
        stats = bvs.step()
        after_energy = sum(node.energy for node in env.nodes)

        tp += stats["tp"]
        tn += stats["tn"]
        fp += stats["fp"]
        fn += stats["fn"]
        total_energy_spent += max(0.0, before_energy - after_energy)

        steps_survived += 1

        if env.network_is_failed():
            break

    cls = compute_classification_metrics(tp, tn, fp, fn)

    return {
        "mttf_steps": float(steps_survived),
        "accuracy": cls["accuracy"],
        "fpr": cls["fpr"],
        "fnr": cls["fnr"],
        "precision": cls["precision"],
        "recall": cls["recall"],
        "energy": total_energy_spent,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def run_bvs_monte_carlo(cfg: SimulationConfig) -> Dict[str, float]:
    seeds = [(cfg.seed or 0) + i for i in range(cfg.runs)]
    results: List[Dict[str, float]] = [run_single_bvs_simulation(cfg, s) for s in seeds]

    def avg(key: str) -> float:
        return statistics.mean(r[key] for r in results)

    def std(key: str) -> float:
        values = [r[key] for r in results]
        return statistics.pstdev(values) if len(values) > 1 else 0.0

    summary = {
        "runs": float(cfg.runs),
        "mttf_mean": avg("mttf_steps"),
        "mttf_std": std("mttf_steps"),
        "accuracy_mean": avg("accuracy"),
        "accuracy_std": std("accuracy"),
        "fpr_mean": avg("fpr"),
        "fnr_mean": avg("fnr"),
        "precision_mean": avg("precision"),
        "recall_mean": avg("recall"),
        "energy_mean": avg("energy"),
        "energy_std": std("energy"),
    }

    return summary