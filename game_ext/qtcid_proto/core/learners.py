from __future__ import annotations

from typing import Optional
import random


class QLearner:
    def __init__(
        self,
        actions: tuple[str, ...],
        alpha: float,
        gamma: float,
        epsilon: float,
        seed: Optional[int] = None,
    ) -> None:
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q: dict[str, dict[str, float]] = {}

    def _ensure(self, state: str) -> None:
        if state not in self.q:
            self.q[state] = {a: 0.0 for a in self.actions}

    def select(self, state: str) -> str:
        self._ensure(state)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        best = max(self.q[state].values())
        actions = [a for a, v in self.q[state].items() if v == best]
        return self.rng.choice(actions)

    def update(self, state: str, action: str, reward: float, next_state: str) -> None:
        self._ensure(state)
        self._ensure(next_state)
        current = self.q[state][action]
        next_best = max(self.q[next_state].values())
        self.q[state][action] = current + self.alpha * (reward + self.gamma * next_best - current)

    def decay(self, decay: float = 0.997, min_epsilon: float = 0.05) -> None:
        self.epsilon = max(min_epsilon, self.epsilon * decay)