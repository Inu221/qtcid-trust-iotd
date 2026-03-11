from __future__ import annotations

from pathlib import Path
import json

from game_ext.qtcid_proto.core.config import WangCycleConfig
from game_ext.qtcid_proto.core.simulator import (
    run_bvs,
    run_soid,
    run_qtcid,
    run_bvs_energy,
    run_soid_energy,
    run_qtcid_energy,
)


def make_cfg(pa: float, tids: int, lam: float, runs: int = 120) -> WangCycleConfig:
    return WangCycleConfig(
        pa=pa,
        tids=tids,
        lambda_capture=lam,
        runs=runs,
        pc=0.5,
        beta=1.0,
    )


def make_cfg_bvs(pa: float, tids: int, lam: float, runs: int = 120) -> WangCycleConfig:
    return WangCycleConfig(
        pa=pa,
        tids=tids,
        lambda_capture=lam,
        runs=runs,
        pc=0.0,
        beta=1.0,
    )


def main() -> None:
    lambdas = [1 / 3600, 1 / 1800, 1 / 900, 1 / 500]
    pa_values = [0.5, 1.0]
    tids = 200

    rows = []

    for pa in pa_values:
        for lam in lambdas:
            cfg_bvs = make_cfg_bvs(pa=pa, tids=tids, lam=lam)
            cfg_soid = make_cfg(pa=pa, tids=tids, lam=lam)
            cfg_qtcid = make_cfg(pa=pa, tids=tids, lam=lam)

            bvs = run_bvs(cfg_bvs)
            soid = run_soid(cfg_soid)
            qtcid = run_qtcid(cfg_qtcid)

            bvs_e = run_bvs_energy(cfg_bvs)
            soid_e = run_soid_energy(cfg_soid)
            qtcid_e = run_qtcid_energy(cfg_qtcid)

            row = {
                "pa": pa,
                "lambda_capture": lam,

                "bvs_mttf": bvs["mttf_mean"],
                "soid_mttf": soid["mttf_mean"],
                "qtcid_mttf": qtcid["mttf_mean"],

                "bvs_acc": bvs["accuracy_mean"],
                "soid_acc": soid["accuracy_mean"],
                "qtcid_acc": qtcid["accuracy_mean"],

                "bvs_energy": bvs_e["energy_spent_mean"],
                "soid_energy": soid_e["energy_spent_mean"],
                "qtcid_energy": qtcid_e["energy_spent_mean"],

                "bvs_audit_energy": bvs_e["audit_energy_mean"],
                "soid_audit_energy": soid_e["audit_energy_mean"],
                "qtcid_audit_energy": qtcid_e["audit_energy_mean"],

                "bvs_audits": bvs["audits_mean"],
                "soid_audits": soid["audits_mean"],
                "qtcid_audits": qtcid["audits_mean"],
            }
            rows.append(row)

            print(
                f"Pa={pa:.2f}, lambda={lam:.6f} | "
                f"MTTF: BVS={bvs['mttf_mean']:.1f}, SOID={soid['mttf_mean']:.1f}, Q={qtcid['mttf_mean']:.1f} | "
                f"ACC: BVS={bvs['accuracy_mean']:.4f}, SOID={soid['accuracy_mean']:.4f}, Q={qtcid['accuracy_mean']:.4f} | "
                f"E: BVS={bvs_e['energy_spent_mean']:.1f}, SOID={soid_e['energy_spent_mean']:.1f}, Q={qtcid_e['energy_spent_mean']:.1f}"
            )

    Path("results").mkdir(exist_ok=True)
    with open("results/diag_capture_sweep.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()