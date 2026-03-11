from __future__ import annotations

from .config import WangCycleConfig
from .utils import safe_div
from .learners import QLearner


class BasePolicy:
    def host_params(self, sim) -> tuple[float, float, float]:
        return sim.cfg.hpfp, sim.cfg.hpfn, 1.0

    def handle_placeholder(self, sim, layer_idx: int, remaining_fn: int, new_layers: list[int]) -> None:
        audited_count = remaining_fn
        sim.audits += audited_count
        spent = audited_count * sim.audit_energy_value
        sim.energy_left -= spent
        sim.audit_energy_total += spent

        attacked_and_audited = sim.binomial(remaining_fn, sim.cfg.pa)
        audited_not_attacked = audited_count - attacked_and_audited

        if layer_idx == len(sim.layers) - 1:
            sim.ne += attacked_and_audited
            sim.tp += attacked_and_audited
        else:
            new_layers[layer_idx + 1] += attacked_and_audited
            sim.fn += attacked_and_audited

        new_layers[layer_idx] += audited_not_attacked
        sim.fn += audited_not_attacked


class SOIDPolicy(BasePolicy):
    def handle_placeholder(self, sim, layer_idx: int, remaining_fn: int, new_layers: list[int]) -> None:
        fixed_check_prob = 0.25 + 0.20 * min(1.0, sim.cfg.pa)
        for _ in range(remaining_fn):
            attacked = sim.rng.random() < sim.cfg.pa
            do_check = sim.rng.random() < fixed_check_prob

            if do_check:
                sim.audits += 1
                sim.energy_left -= sim.audit_energy_value
                sim.audit_energy_total += sim.audit_energy_value
                if attacked:
                    if layer_idx == len(sim.layers) - 1:
                        sim.ne += 1
                    else:
                        new_layers[layer_idx + 1] += 1
                    sim.tp += 1
                else:
                    new_layers[layer_idx] += 1
                    sim.fn += 1
            else:
                new_layers[layer_idx] += 1
                sim.fn += 1


class QTCIDPolicy(SOIDPolicy):
    def __init__(self, cfg: WangCycleConfig, seed: int) -> None:
        self.cfg = cfg
        self.audit_agent = QLearner(
            ("check", "trust"),
            cfg.alpha_s,
            cfg.gamma_s,
            cfg.epsilon_s,
            seed + 10000,
        )
        self.host_agent = QLearner(
            ("balanced", "secure", "aggressive"),
            cfg.alpha_n,
            cfg.gamma_n,
            cfg.epsilon_n,
            seed + 20000,
        )
        self._host_state = "low|low|mid"
        self._host_action = "balanced"

    def _host_state_repr(self, sim) -> str:
        bad_ratio = safe_div(sim.nb_total(), max(1, sim.active_total()))
        energy_ratio = safe_div(sim.energy_left, sim.cfg.initial_system_energy)

        risk = "low" if bad_ratio < 0.10 else "mid" if bad_ratio < 0.20 else "high"
        pa_band = "low" if sim.cfg.pa < 0.34 else "mid" if sim.cfg.pa < 0.67 else "high"
        energy_band = "low" if energy_ratio < 0.35 else "mid" if energy_ratio < 0.70 else "high"
        return f"{risk}|{pa_band}|{energy_band}"

    def host_params(self, sim) -> tuple[float, float, float]:
        self._host_state = self._host_state_repr(sim)
        self._host_action = self.host_agent.select(self._host_state)
        if self._host_action == "balanced":
            return sim.cfg.hpfp * 0.90, sim.cfg.hpfn * 0.82, 0.94
        if self._host_action == "secure":
            return sim.cfg.hpfp * 0.82, sim.cfg.hpfn * 0.62, 0.95
        return sim.cfg.hpfp * 0.78, sim.cfg.hpfn * 0.68, 0.97

    def finalize_host_learning(self, sim, delta_tp: int, delta_tn: int, delta_fp: int, delta_fn: int, energy_used: float) -> None:
        next_state = self._host_state_repr(sim)
        reward = (
            2.8 * delta_tp
            + 1.0 * delta_tn
            - 3.2 * delta_fp
            - 4.2 * delta_fn
            - 0.65 * safe_div(energy_used, sim.vote_energy_value)
        )
        self.host_agent.update(self._host_state, self._host_action, reward, next_state)
        self.host_agent.decay()

    def handle_placeholder(self, sim, layer_idx: int, remaining_fn: int, new_layers: list[int]) -> None:
        for _ in range(remaining_fn):
            state = sim.audit_state(layer_idx)
            action = self.audit_agent.select(state)
            attacked = sim.rng.random() < sim.cfg.pa

            if action == "check":
                sim.audits += 1
                sim.energy_left -= sim.audit_energy_value
                sim.audit_energy_total += sim.audit_energy_value
                if attacked:
                    if layer_idx == len(sim.layers) - 1:
                        sim.ne += 1
                    else:
                        new_layers[layer_idx + 1] += 1
                    sim.tp += 1
                else:
                    new_layers[layer_idx] += 1
                    sim.fn += 1
            else:
                new_layers[layer_idx] += 1
                sim.fn += 1

            next_state = sim.audit_state(min(layer_idx + 1, len(sim.layers) - 1) if action == "check" and attacked else layer_idx)
            audit_cost_norm = safe_div(sim.audit_energy_value, sim.vote_energy_value)
            if action == "check" and attacked:
                reward = sim.cfg.reward_b - 1.5 * audit_cost_norm
            elif action == "trust" and attacked:
                reward = -sim.cfg.reward_b
            elif action == "check" and not attacked:
                reward = -2.2 * audit_cost_norm
            else:
                reward = 0.35

            self.audit_agent.update(state, action, reward, next_state)

        self.audit_agent.decay()