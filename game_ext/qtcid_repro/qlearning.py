from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass
class QActionSelection:
    state: str
    action: str
    explored: bool


class QTableLearner:
    def __init__(
        self,
        actions: tuple[str, ...],
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_decay: float = 0.997,
        epsilon_min: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = random.Random(seed)
        self.q: Dict[str, Dict[str, float]] = {}

    def _ensure_state(self, state: str) -> None:
        if state not in self.q:
            self.q[state] = {a: 0.0 for a in self.actions}

    def greedy_action(self, state: str) -> str:
        self._ensure_state(state)
        q_state = self.q[state]
        best = max(q_state.values())
        best_actions = [a for a, v in q_state.items() if v == best]
        return self.rng.choice(best_actions)

    def select_action(self, state: str) -> QActionSelection:
        self._ensure_state(state)

        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
            return QActionSelection(state=state, action=action, explored=True)

        action = self.greedy_action(state)
        return QActionSelection(state=state, action=action, explored=False)

    def update(self, state: str, action: str, reward: float, next_state: str) -> None:
        self._ensure_state(state)
        self._ensure_state(next_state)

        current = self.q[state][action]
        next_best = max(self.q[next_state].values())

        self.q[state][action] = current + self.alpha * (
            reward + self.gamma * next_best - current
        )

    def decay(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def q_value(self, state: str, action: str) -> float:
        self._ensure_state(state)
        return self.q[state][action]

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {s: dict(v) for s, v in self.q.items()}