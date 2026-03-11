from __future__ import annotations

from pathlib import Path
import json

from game_ext.qtcid_proto.core.config import make_bvs_config, make_soid_config, make_qtcid_config
from game_ext.qtcid_proto.core.simulator import run_bvs, run_soid, run_qtcid


def main() -> None:
    cfg_bvs = make_bvs_config(pa=0.5, tids=200, runs=100)
    cfg_soid = make_soid_config(pa=0.5, tids=200, runs=100)
    cfg_qtcid = make_qtcid_config(pa=0.5, tids=200, runs=100)

    bvs = run_bvs(cfg_bvs)
    soid = run_soid(cfg_soid)
    qtcid = run_qtcid(cfg_qtcid)

    Path("results").mkdir(exist_ok=True)
    for name, data in [("bvs", bvs), ("soid", soid), ("qtcid", qtcid)]:
        with open(f"results/{name}_single.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print("BVS", bvs)
    print("SOID", soid)
    print("Q-TCID", qtcid)


if __name__ == "__main__":
    main()