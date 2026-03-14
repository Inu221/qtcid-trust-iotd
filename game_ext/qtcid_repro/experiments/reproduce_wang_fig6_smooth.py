from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from game_ext.qtcid_repro.wang.bvs_core import (
    WangBVSConfig,
    run_wang_bvs_monte_carlo,
)


def moving_average(values: list[float], window: int = 3) -> list[float]:
    if window <= 1 or len(values) < window:
        return values[:]

    half = window // 2
    out = []
    n = len(values)

    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        out.append(sum(values[left:right]) / (right - left))
    return out


def main() -> None:
    pa_values = [0.55, 0.60, 0.65, 0.70, 0.75]
    pc_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    curves_raw = []
    curves_smooth = []

    for pa in pa_values:
        y = []
        for pc in pc_values:
            cfg = WangBVSConfig(
                n_nodes=128,
                n_neighbors=32,
                m_voters=5,
                tids=320,
                lambda_capture=1 / 1800,
                hpfn=0.10,
                hpfp=0.10,
                pa=pa,
                pc=pc,
                beta=1.0,
                runs=300,
                max_time=1000000,
                seed=42,
            )
            stats = run_wang_bvs_monte_carlo(cfg)
            y.append(stats.mttf_mean)
            print(f"Pa={pa:.2f}, Pc={pc:.2f}, MTTF={stats.mttf_mean:.2f}")

        curves_raw.append((pa, y))
        curves_smooth.append((pa, moving_average(y, window=3)))

    Path("results").mkdir(exist_ok=True)

    plt.figure(figsize=(9, 6))
    for pa, y in curves_raw:
        plt.plot(pc_values, y, marker="o", markersize=4, linewidth=1.0, label=f"Pa = {pa:.2f}")
    plt.xlabel("Pc")
    plt.ylabel("MTTF")
    plt.title("Wang Figure 6 reproduction (raw points)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/reproduce_wang_fig6_from_bvs_raw.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 6))
    for pa, y in curves_smooth:
        plt.plot(pc_values, y, linewidth=2.0, label=f"Pa = {pa:.2f}")
    plt.xlabel("Pc")
    plt.ylabel("MTTF")
    plt.title("Wang Figure 6 reproduction: MTTF versus Pc")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/reproduce_wang_fig6_from_bvs_smooth.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    main()