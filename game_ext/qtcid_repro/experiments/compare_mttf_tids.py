from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from game_ext.qtcid_repro.wang.bvs_core import WangBVSConfig, run_wang_bvs_monte_carlo
from game_ext.qtcid_repro.soid_core import SOIDConfig, run_soid_monte_carlo


def compare_wang_soid_mttf_tids() -> None:
    pa_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    tids_values = list(range(50, 1601, 150))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, pa in enumerate(pa_values):
        wang_y = []
        soid_y = []

        for tids in tids_values:
            wang_cfg = WangBVSConfig(
                pa=pa,
                pc=0.5,
                beta=1.0,
                tids=tids,
                runs=80,
                max_time=16000,
                seed=42,
            )
            soid_cfg = SOIDConfig(
                pa=pa,
                pc=0.5,
                beta=1.0,
                tids=tids,
                runs=80,
                max_time=16000,
                seed=42,
            )

            wang_res = run_wang_bvs_monte_carlo(wang_cfg)
            soid_res = run_soid_monte_carlo(soid_cfg)

            wang_y.append(wang_res.mttf_mean)
            soid_y.append(soid_res.mttf_mean)

            print(
                f"Pa={pa:.2f}, TIDS={tids} | "
                f"Wang-BVS={wang_res.mttf_mean:.2f}, SOID={soid_res.mttf_mean:.2f}"
            )

        ax = axes[idx]
        ax.plot(tids_values, wang_y, marker="o", linewidth=1.7, markersize=3.5, label="BVS (Wang)")
        ax.plot(tids_values, soid_y, marker="o", linewidth=1.7, markersize=3.5, label="SOID")
        ax.set_title(f"Pa = {pa}")
        ax.set_xlabel("Интервал IDS TIDS")
        ax.set_ylabel("Среднее время до отказа MTTF")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.delaxes(axes[-1])
    fig.tight_layout()

    Path("results").mkdir(exist_ok=True)
    fig.savefig("results/qtcid_repro_wang_soid_mttf_tids.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    compare_wang_soid_mttf_tids()