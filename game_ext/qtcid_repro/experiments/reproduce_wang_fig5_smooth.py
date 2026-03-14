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
    m_values = [3, 5, 7]
    tids_values = [5, 10, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]

    curves_raw = []
    curves_smooth = []

    for m in m_values:
        y = []
        for tids in tids_values:
            cfg = WangBVSConfig(
                n_nodes=128,
                n_neighbors=32,
                m_voters=m,
                tids=tids,
                lambda_capture=1 / 3600,
                hpfn=0.10,
                hpfp=0.10,
                pa=0.5,
                pc=0.5,
                beta=1.0,
                runs=300,
                max_time=1000000,
                seed=42,
            )
            stats = run_wang_bvs_monte_carlo(cfg)
            y.append(stats.mttf_mean)
            print(f"m={m}, TIDS={tids}, MTTF={stats.mttf_mean:.2f}")

        curves_raw.append((m, y))
        curves_smooth.append((m, moving_average(y, window=3)))

    Path("results").mkdir(exist_ok=True)

    plt.figure(figsize=(9, 6))
    for m, y in curves_raw:
        plt.plot(tids_values, y, marker="o", markersize=4, linewidth=1.0, label=f"m = {m}")
    plt.xlabel("TIDS")
    plt.ylabel("MTTF")
    plt.title("Wang Figure 5 reproduction (raw points)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/reproduce_wang_fig5_from_bvs_raw.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 6))
    for m, y in curves_smooth:
        plt.plot(tids_values, y, linewidth=2.0, label=f"m = {m}")
    plt.xlabel("TIDS")
    plt.ylabel("MTTF")
    plt.title("Wang Figure 5 reproduction: MTTF versus TIDS")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/reproduce_wang_fig5_from_bvs_smooth.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    main()