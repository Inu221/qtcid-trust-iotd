from __future__ import annotations

from game_ext.qtcid_repro.mitchell.core import MitchellConfig, run_mitchell_monte_carlo
from game_ext.qtcid_repro.types import QTCIDConfig
from game_ext.qtcid_repro.qtcid.core import run_qtcid_monte_carlo


def main() -> None:
    mcfg = MitchellConfig(
        runs=10,
        max_time=2000,
        tids=200,
        seed=42,
    )
    mres = run_mitchell_monte_carlo(mcfg)
    print("MITCHELL")
    print(mres)

    qcfg = QTCIDConfig(
        n_mc_runs=10,
        max_time=1000,
        tids=200,
        seed=42,
    )
    qres = run_qtcid_monte_carlo(qcfg)
    print("Q-TCID")
    print(qres)


if __name__ == "__main__":
    main()