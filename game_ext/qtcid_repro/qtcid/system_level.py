from __future__ import annotations

from dataclasses import dataclass

from game_ext.qtcid_repro.qlearning import QTableLearner
from game_ext.qtcid_repro.types import QTCIDConfig
from game_ext.qtcid_repro.utils import safe_div


SYSTEM_ACTIONS = ("check", "trust")


@dataclass
class SystemAuditDecision:
    state: str
    action: str
    reward: float
    next_state: str
    attacked: bool
    explored: bool


class SystemLevelAuditAgent:
    """
    Faithful to article at system level:
    - actions AC = {check, trust}
    - reward function exactly from Eq. (9):
        B - Ea    if check, attack
        -B        if trust, attack
        -Ea       if check, silent
        0         if trust, silent
    - epsilon-greedy as Eq. (8)
    - Q-update as Eq. (10)
    """

    def __init__(self, cfg: QTCIDConfig) -> None:
        self.cfg = cfg
        self.learner = QTableLearner(
            actions=SYSTEM_ACTIONS,
            alpha=cfg.system_q.alpha,
            gamma=cfg.system_q.gamma,
            epsilon=cfg.system_q.epsilon,
            epsilon_decay=cfg.system_q.epsilon_decay,
            epsilon_min=cfg.system_q.epsilon_min,
            seed=cfg.seed + 20000,
        )

    def build_state(
        self,
        bad_ratio: float,
        pa: float,
        audit_budget_ratio: float,
    ) -> str:
        bad_band = "low" if bad_ratio < 0.10 else "mid" if bad_ratio < 0.20 else "high"
        pa_band = "low" if pa < 0.34 else "mid" if pa < 0.67 else "high"
        budget_band = (
            "low" if audit_budget_ratio < 0.25
            else "mid" if audit_budget_ratio < 0.60
            else "high"
        )
        return f"{bad_band}|{pa_band}|{budget_band}"

    def reward(self, action: str, attacked: bool) -> float:
        b = self.cfg.reward_b
        ea = self.cfg.audit_cost

        if action == "check" and attacked:
            return b - ea
        if action == "trust" and attacked:
            return -b
        if action == "check" and not attacked:
            return -ea
        return 0.0

    def choose_action(
        self,
        bad_ratio: float,
        audit_budget_ratio: float,
    ) -> tuple[str, str, bool]:
        state = self.build_state(
            bad_ratio=bad_ratio,
            pa=self.cfg.pa,
            audit_budget_ratio=audit_budget_ratio,
        )
        sel = self.learner.select_action(state)
        return state, sel.action, sel.explored

    def step(
        self,
        bad_ratio: float,
        audit_budget_ratio: float,
        attacked: bool,
        next_bad_ratio: float,
        next_audit_budget_ratio: float,
    ) -> SystemAuditDecision:
        state, action, explored = self.choose_action(
            bad_ratio=bad_ratio,
            audit_budget_ratio=audit_budget_ratio,
        )

        r = self.reward(action, attacked)

        next_state = self.build_state(
            bad_ratio=next_bad_ratio,
            pa=self.cfg.pa,
            audit_budget_ratio=next_audit_budget_ratio,
        )

        self.learner.update(state, action, r, next_state)
        self.learner.decay()

        return SystemAuditDecision(
            state=state,
            action=action,
            reward=r,
            next_state=next_state,
            attacked=attacked,
            explored=explored,
        )