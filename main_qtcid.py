from __future__ import annotations

import json
from pathlib import Path

from game_ext.qtcid.plots import plot_curve
from game_ext.qtcid.simulator import SimulationConfig, run_bvs_monte_carlo


def experiment_bvs_single() -> None:
    cfg = SimulationConfig()
    summary = run_bvs_monte_carlo(cfg)

    Path("results").mkdir(exist_ok=True)

    with open("results/bvs_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("BVS summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


def experiment_bvs_mttf_vs_pa() -> None:
    pa_values = [0.0, 0.1, 0.2, 0.3, 0.4]
    mttf_values = []

    for pa in pa_values:
        cfg = SimulationConfig(
            malicious_vote_probability=pa,
            runs=20,
        )
        summary = run_bvs_monte_carlo(cfg)
        mttf_values.append(summary["mttf_mean"])
        print(f"Pa={pa:.2f} -> MTTF={summary['mttf_mean']:.3f}")

    plot_curve(
        x=pa_values,
        y=mttf_values,
        xlabel="Pa",
        ylabel="MTTF (steps)",
        title="BVS: MTTF vs Pa",
        save_path="results/bvs_mttf_vs_pa.png",
    )


if __name__ == "__main__":
    experiment_bvs_single()
    experiment_bvs_mttf_vs_pa()