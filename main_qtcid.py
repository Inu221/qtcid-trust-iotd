from __future__ import annotations

import json
from pathlib import Path

from game_ext.qtcid_proto.plots import plot_multi_curve, plot_mttf_panels_by_pa
from game_ext.qtcid_proto.simulator import SimulationConfig, run_monte_carlo


def compare_accuracy_vs_pa() -> None:
    pa_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    bvs_vals = []
    soid_vals = []
    qtcid_vals = []

    for pa in pa_values:
        cfg = SimulationConfig(malicious_vote_probability=pa, runs=20)

        bvs = run_monte_carlo("bvs", cfg, keep_first_history=False)
        soid = run_monte_carlo("soid", cfg, keep_first_history=False)
        qtcid = run_monte_carlo("qtcid", cfg, keep_first_history=False)

        bvs_vals.append(bvs["accuracy_mean"])
        soid_vals.append(soid["accuracy_mean"])
        qtcid_vals.append(qtcid["accuracy_mean"])

        print(
            f"Pa={pa:.2f} | "
            f"BVS={bvs['accuracy_mean']:.4f} | "
            f"SOID={soid['accuracy_mean']:.4f} | "
            f"Q-TCID={qtcid['accuracy_mean']:.4f}"
        )

    plot_multi_curve(
        x=pa_values,
        series=[
            {"label": "BVS", "y": bvs_vals},
            {"label": "SOID", "y": soid_vals},
            {"label": "Q-TCID", "y": qtcid_vals},
        ],
        xlabel="Вероятность атаки Pa",
        ylabel="Точность обнаружения",
        title="Сравнение точности обнаружения",
        save_path="results/compare_accuracy_vs_pa_ru.png",
    )


def compare_energy_vs_pa() -> None:
    pa_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    bvs_vals = []
    soid_vals = []
    qtcid_vals = []

    for pa in pa_values:
        cfg = SimulationConfig(malicious_vote_probability=pa, runs=20)

        bvs = run_monte_carlo("bvs", cfg, keep_first_history=False)
        soid = run_monte_carlo("soid", cfg, keep_first_history=False)
        qtcid = run_monte_carlo("qtcid", cfg, keep_first_history=False)

        bvs_vals.append(bvs["energy_mean"])
        soid_vals.append(soid["energy_mean"])
        qtcid_vals.append(qtcid["energy_mean"])

        print(
            f"Pa={pa:.2f} | "
            f"BVS={bvs['energy_mean']:.4f} | "
            f"SOID={soid['energy_mean']:.4f} | "
            f"Q-TCID={qtcid['energy_mean']:.4f}"
        )

    plot_multi_curve(
        x=pa_values,
        series=[
            {"label": "BVS", "y": bvs_vals},
            {"label": "SOID", "y": soid_vals},
            {"label": "Q-TCID", "y": qtcid_vals},
        ],
        xlabel="Вероятность атаки Pa",
        ylabel="Энергопотребление E",
        title="Сравнение энергопотребления",
        save_path="results/compare_energy_vs_pa_ru.png",
    )


def compare_mttf_figure3_style() -> None:
    pa_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    tids_values = list(range(50, 1601, 150))
    panels = []

    for pa in pa_values:
        bvs_vals = []
        soid_vals = []
        qtcid_vals = []

        for tids in tids_values:
            cfg = SimulationConfig(
                malicious_vote_probability=pa,
                tids=tids,
                runs=20,
            )
            bvs = run_monte_carlo("bvs", cfg, keep_first_history=False)
            soid = run_monte_carlo("soid", cfg, keep_first_history=False)
            qtcid = run_monte_carlo("qtcid", cfg, keep_first_history=False)

            bvs_vals.append(bvs["mttf_mean"])
            soid_vals.append(soid["mttf_mean"])
            qtcid_vals.append(qtcid["mttf_mean"])

            print(
                f"Pa={pa:.2f}, TIDS={tids} | "
                f"BVS={bvs['mttf_mean']:.2f} | "
                f"SOID={soid['mttf_mean']:.2f} | "
                f"Q-TCID={qtcid['mttf_mean']:.2f}"
            )

        panels.append(
            {
                "title": f"Pa = {pa}",
                "series": [
                    {"label": "BVS", "y": bvs_vals},
                    {"label": "SOID", "y": soid_vals},
                    {"label": "Q-TCID", "y": qtcid_vals},
                ],
            }
        )

    plot_mttf_panels_by_pa(
        tids_values=tids_values,
        panels=panels,
        save_path="results/compare_figure3_bvs_soid_qtcid_ru.png",
    )


def dump_single_runs() -> None:
    Path("results").mkdir(exist_ok=True)

    cfg = SimulationConfig()
    data = {
        "bvs": run_monte_carlo("bvs", cfg),
        "soid": run_monte_carlo("soid", cfg),
        "qtcid": run_monte_carlo("qtcid", cfg),
    }

    with open("results/bvs_soid_qtcid_summary.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    dump_single_runs()
    compare_accuracy_vs_pa()
    compare_energy_vs_pa()
    compare_mttf_figure3_style()