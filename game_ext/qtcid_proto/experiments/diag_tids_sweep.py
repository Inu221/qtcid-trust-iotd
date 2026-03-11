from __future__ import annotations

from pathlib import Path
import json

from game_ext.qtcid_proto.core.config import make_bvs_config, make_soid_config, make_qtcid_config
from game_ext.qtcid_proto.core.simulator import (
    run_bvs,
    run_soid,
    run_qtcid,
    run_bvs_energy,
    run_soid_energy,
    run_qtcid_energy,
)


def main() -> None:
    tids_values = [50, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1400]
    pa_values = [0.0, 0.5, 1.0]

    rows = []

    for pa in pa_values:
        for tids in tids_values:
            cfg_bvs = make_bvs_config(pa=pa, tids=tids, runs=120)
            cfg_soid = make_soid_config(pa=pa, tids=tids, runs=120)
            cfg_qtcid = make_qtcid_config(pa=pa, tids=tids, runs=120)

            bvs = run_bvs(cfg_bvs)
            soid = run_soid(cfg_soid)
            qtcid = run_qtcid(cfg_qtcid)

            bvs_e = run_bvs_energy(cfg_bvs)
            soid_e = run_soid_energy(cfg_soid)
            qtcid_e = run_qtcid_energy(cfg_qtcid)

            row = {
                "pa": pa,
                "tids": tids,

                "bvs_mttf": bvs["mttf_mean"],
                "soid_mttf": soid["mttf_mean"],
                "qtcid_mttf": qtcid["mttf_mean"],

                "bvs_acc": bvs["accuracy_mean"],
                "soid_acc": soid["accuracy_mean"],
                "qtcid_acc": qtcid["accuracy_mean"],

                "bvs_energy": bvs_e["energy_spent_mean"],
                "soid_energy": soid_e["energy_spent_mean"],
                "qtcid_energy": qtcid_e["energy_spent_mean"],

                "bvs_vote_energy": bvs_e["vote_energy_mean"],
                "bvs_audit_energy": bvs_e["audit_energy_mean"],
                "soid_vote_energy": soid_e["vote_energy_mean"],
                "soid_audit_energy": soid_e["audit_energy_mean"],
                "qtcid_vote_energy": qtcid_e["vote_energy_mean"],
                "qtcid_audit_energy": qtcid_e["audit_energy_mean"],

                "bvs_audits": bvs["audits_mean"],
                "soid_audits": soid["audits_mean"],
                "qtcid_audits": qtcid["audits_mean"],
            }
            rows.append(row)

            print(
                f"Pa={pa:.2f}, TIDS={tids} | "
                f"MTTF: BVS={bvs['mttf_mean']:.1f}, SOID={soid['mttf_mean']:.1f}, Q={qtcid['mttf_mean']:.1f} | "
                f"E: BVS={bvs_e['energy_spent_mean']:.1f}, SOID={soid_e['energy_spent_mean']:.1f}, Q={qtcid_e['energy_spent_mean']:.1f}"
            )

    Path("results").mkdir(exist_ok=True)
    with open("results/diag_tids_sweep.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()