from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from game_ext.qtcid_proto.core.config import make_bvs_config, make_soid_config, make_qtcid_config
from game_ext.qtcid_proto.core.simulator import (
    run_bvs,
    run_soid,
    run_qtcid,
    run_bvs_energy,
    run_soid_energy,
    run_qtcid_energy,
)


def compare_accuracy_energy() -> None:
    pa_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    bvs_acc = []
    soid_acc = []
    qtcid_acc = []

    bvs_e = []
    soid_e = []
    qtcid_e = []

    for pa in pa_values:
        cfg_bvs = make_bvs_config(pa=pa, tids=200, runs=1000)
        cfg_soid = make_soid_config(pa=pa, tids=200, runs=1000)
        cfg_qtcid = make_qtcid_config(pa=pa, tids=200, runs=1000)

        cfg_bvs.hpfp = 0.10
        cfg_bvs.hpfn = 0.10

        cfg_soid.hpfp = 0.10
        cfg_soid.hpfn = 0.10

        cfg_qtcid.hpfp = 0.10
        cfg_qtcid.hpfn = 0.10

        cfg_bvs.audit_cost_scale = 25.0
        cfg_soid.audit_cost_scale = 25.0
        cfg_qtcid.audit_cost_scale = 25.0

        cfg_bvs.lambda_capture = 1 / 1800
        cfg_soid.lambda_capture = 1 / 1800
        cfg_qtcid.lambda_capture = 1 / 1800
        
        bvs = run_bvs(cfg_bvs)
        soid = run_soid(cfg_soid)
        qtcid = run_qtcid(cfg_qtcid)

        bvs_energy = run_bvs_energy(cfg_bvs)
        soid_energy = run_soid_energy(cfg_soid)
        qtcid_energy = run_qtcid_energy(cfg_qtcid)

        bvs_acc.append(bvs["accuracy_mean"])
        soid_acc.append(soid["accuracy_mean"])
        qtcid_acc.append(qtcid["accuracy_mean"])

        bvs_e.append(bvs_energy["energy_spent_mean"])
        soid_e.append(soid_energy["energy_spent_mean"])
        qtcid_e.append(qtcid_energy["energy_spent_mean"])

        print(
            f"Pa={pa:.2f}, "
            f"BVS acc={bvs['accuracy_mean']:.4f}, "
            f"SOID acc={soid['accuracy_mean']:.4f}, "
            f"Q-TCID acc={qtcid['accuracy_mean']:.4f}, "
            f"BVS E={bvs_energy['energy_spent_mean']:.2f}, "
            f"SOID E={soid_energy['energy_spent_mean']:.2f}, "
            f"Q-TCID E={qtcid_energy['energy_spent_mean']:.2f}"
        )

    Path("results").mkdir(exist_ok=True)

    x = range(len(pa_values))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar([i - width for i in x], bvs_acc, width=width, label="BVS")
    plt.bar(x, soid_acc, width=width, label="SOID")
    plt.bar([i + width for i in x], qtcid_acc, width=width, label="Q-TCID")
    plt.xticks(list(x), [str(v) for v in pa_values])
    plt.xlabel("Вероятность атаки, Pa")
    plt.ylabel("Точность обнаружения")
    plt.title("Сравнение точности обнаружения")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/graph_compare_accuracy_bar.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar([i - width for i in x], bvs_e, width=width, label="BVS")
    plt.bar(x, soid_e, width=width, label="SOID")
    plt.bar([i + width for i in x], qtcid_e, width=width, label="Q-TCID")
    plt.xticks(list(x), [str(v) for v in pa_values])
    plt.xlabel("Вероятность атаки, Pa")
    plt.ylabel("Энергопотребление, E")
    plt.title("Сравнение энергопотребления")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/graph_compare_energy_bar.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    compare_accuracy_energy()