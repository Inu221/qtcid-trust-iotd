from __future__ import annotations

from game_ext.qtcid_repro.wang.bvs_core import WangBVSConfig, run_wang_bvs_monte_carlo
from game_ext.qtcid_repro.wang.game import pmin_c, theorem1_attack_discouraged


def main() -> None:
    beta = 1.0
    pc = pmin_c(beta)

    print("beta =", beta)
    print("Pc_min =", pc)
    print("Attack discouraged:", theorem1_attack_discouraged(pc, beta))

    cfg = WangBVSConfig(
        beta=beta,
        pc=pc,
        pa=0.5,
        tids=200,
        runs=10,
        max_time=3000,
        seed=42,
    )

    res = run_wang_bvs_monte_carlo(cfg)
    print("WANG-BVS")
    print(res)


if __name__ == "__main__":
    main()