from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from game_ext.qtcid_repro.wang.bvs_core import WangBVSConfig, run_wang_bvs_monte_carlo
from game_ext.qtcid_repro.soid_core import SOIDConfig, run_soid_monte_carlo
from game_ext.qtcid_repro.qtcid_enhanced_core import QTCIDEnhancedConfig, run_qtcid_enhanced_monte_carlo
from game_ext.qtcid_repro.qtcid_trust_core import QTCIDTrustConfig, run_qtcid_trust_monte_carlo


def main() -> None:
    pa_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    wang_acc, soid_acc, q_acc, trust_acc = [], [], [], []
    wang_e, soid_e, q_e, trust_e = [], [], [], []

    for pa in pa_values:
        wang_cfg = WangBVSConfig(
            pa=pa,
            pc=0.5,
            beta=1.0,
            tids=200,
            runs=120,
            max_time=12000,
            seed=42,
        )
        soid_cfg = SOIDConfig(
            pa=pa,
            pc=0.5,
            beta=1.0,
            tids=200,
            runs=120,
            max_time=12000,
            seed=42,
        )
        q_cfg = QTCIDEnhancedConfig(
            pa=pa,
            pc=0.5,
            beta=1.0,
            tids=200,
            runs=120,
            max_time=12000,
            seed=42,
            reward_b=1.0,
        )
        trust_cfg = QTCIDTrustConfig(
            pa=pa,
            pc=0.5,
            beta=1.0,
            tids=200,
            runs=120,
            max_time=12000,
            seed=42,
            reward_b=1.0,
        )

        wang = run_wang_bvs_monte_carlo(wang_cfg)
        soid = run_soid_monte_carlo(soid_cfg)
        qtcid = run_qtcid_enhanced_monte_carlo(q_cfg)
        trust = run_qtcid_trust_monte_carlo(trust_cfg)

        wang_acc.append(wang.accuracy_mean)
        soid_acc.append(soid.accuracy_mean)
        q_acc.append(qtcid.accuracy_mean)
        trust_acc.append(trust.accuracy_mean)

        wang_e.append(wang.energy_spent_mean)
        soid_e.append(soid.energy_spent_mean)
        q_e.append(qtcid.energy_spent_mean)
        trust_e.append(trust.energy_spent_mean)

        print(
            f"Pa={pa:.2f} | "
            f"ACC: Wang={wang.accuracy_mean:.4f}, SOID={soid.accuracy_mean:.4f}, "
            f"Q-TCID={qtcid.accuracy_mean:.4f}, Q-TCID-Trust={trust.accuracy_mean:.4f} | "
            f"E: Wang={wang.energy_spent_mean:.2f}, SOID={soid.energy_spent_mean:.2f}, "
            f"Q-TCID={qtcid.energy_spent_mean:.2f}, Q-TCID-Trust={trust.energy_spent_mean:.2f}"
        )

    Path("results").mkdir(exist_ok=True)

    x = range(len(pa_values))
    width = 0.20

    plt.figure(figsize=(10, 5))
    plt.bar([i - 1.5 * width for i in x], wang_acc, width=width, label="Wang-BVS")
    plt.bar([i - 0.5 * width for i in x], soid_acc, width=width, label="SOID")
    plt.bar([i + 0.5 * width for i in x], q_acc, width=width, label="Q-TCID")
    plt.bar([i + 1.5 * width for i in x], trust_acc, width=width, label="Q-TCID-Trust")
    plt.xticks(list(x), [str(v) for v in pa_values])
    plt.xlabel("Pa")
    plt.ylabel("Accuracy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/qtcid_repro_accuracy_all.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar([i - 1.5 * width for i in x], wang_e, width=width, label="Wang-BVS")
    plt.bar([i - 0.5 * width for i in x], soid_e, width=width, label="SOID")
    plt.bar([i + 0.5 * width for i in x], q_e, width=width, label="Q-TCID")
    plt.bar([i + 1.5 * width for i in x], trust_e, width=width, label="Q-TCID-Trust")
    plt.xticks(list(x), [str(v) for v in pa_values])
    plt.xlabel("Pa")
    plt.ylabel("Energy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/qtcid_repro_energy_all.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    main()