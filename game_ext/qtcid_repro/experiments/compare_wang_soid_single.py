from __future__ import annotations

from game_ext.qtcid_repro.wang.bvs_core import WangBVSConfig, run_wang_bvs_monte_carlo
from game_ext.qtcid_repro.soid_core import SOIDConfig, run_soid_monte_carlo


def main() -> None:
    wang_cfg = WangBVSConfig(
        pa=0.5,
        pc=0.5,
        beta=1.0,
        tids=200,
        runs=50,
        max_time=8000,
        seed=42,
    )
    soid_cfg = SOIDConfig(
        pa=0.5,
        pc=0.5,
        beta=1.0,
        tids=200,
        runs=50,
        max_time=8000,
        seed=42,
    )

    wang_res = run_wang_bvs_monte_carlo(wang_cfg)
    soid_res = run_soid_monte_carlo(soid_cfg)

    print("WANG-BVS")
    print(wang_res)
    print("SOID")
    print(soid_res)


if __name__ == "__main__":
    main()