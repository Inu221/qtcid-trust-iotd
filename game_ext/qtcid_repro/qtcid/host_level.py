from __future__ import annotations

from dataclasses import dataclass

from game_ext.qtcid_repro.qlearning import QTableLearner
from game_ext.qtcid_repro.types import Node, NodeState, QTCIDConfig
from game_ext.qtcid_repro.utils import clamp01


def host_actions_for_m(m: int) -> tuple[str, ...]:
    """
    В статье AV = (aV1, ..., aV(m+1)).
    Для m=5 это 6 стратегий, соответствующих 6 типам итоговых vote proportions.
    """
    return tuple(f"aV{k}" for k in range(1, m + 2))


@dataclass
class HostVoteDecision:
    voter_id: int
    target_id: int
    state: str
    action: str
    vote_for_good: bool
    reward: float
    next_state: str
    explored: bool


class HostLevelVotingAgent:
    """
    Faithful operationalization Q-TCID host-level.

    Из статьи мы знаем:
    - state space SV = {target good, target bad}
    - action space AV has size m+1
    - reward max if all voters identify target correctly
    - reward min if all voters misjudge target

    Так как статья не даёт полного исполнимого mapping aVt -> individual vote,
    мы задаём article-consistent operationalization:
    - action controls aggressiveness / conservativeness of the vote
      через требуемую уверенность по истории.
    """

    def __init__(self, cfg: QTCIDConfig) -> None:
        self.cfg = cfg
        self.actions = host_actions_for_m(cfg.m_voters)
        self.learner = QTableLearner(
            actions=self.actions,
            alpha=cfg.host_q.alpha,
            gamma=cfg.host_q.gamma,
            epsilon=cfg.host_q.epsilon,
            epsilon_decay=cfg.host_q.epsilon_decay,
            epsilon_min=cfg.host_q.epsilon_min,
            seed=cfg.seed + 10000,
        )

    def build_state(self, voter: Node, target: Node) -> str:
        target_state = "good" if target.state == NodeState.GOOD else "bad"
        bias = voter.historical_bias_toward(target.node_id)

        if bias <= -0.5:
            hist = "strong_bad"
        elif bias < 0.0:
            hist = "weak_bad"
        elif bias == 0.0:
            hist = "neutral"
        elif bias < 0.5:
            hist = "weak_good"
        else:
            hist = "strong_good"

        return f"{target_state}|{hist}"

    def action_threshold(self, action: str) -> float:
        """
        m+1 стратегий преобразуем в шкалу порога:
        ранние aV_k -> более склонны голосовать "bad"
        поздние aV_k -> более склонны голосовать "good"
        """
        idx = self.actions.index(action)
        if len(self.actions) == 1:
            return 0.0
        return -1.0 + 2.0 * idx / (len(self.actions) - 1)

    def malicious_vote(self, target_is_bad: bool, attack: bool) -> bool:
        if attack:
            return True if target_is_bad else False
        return False if target_is_bad else True

    def good_vote_by_history(self, voter: Node, target: Node, action: str) -> bool:
        bias = voter.historical_bias_toward(target.node_id)
        threshold = self.action_threshold(action)
        score = bias - threshold

        # Если score >= 0, узел склонен считать target good
        return score >= 0.0

    def reward_value(
        self,
        target_is_good: bool,
        vote_for_good: bool,
        all_correct: bool,
        all_wrong: bool,
    ) -> float:
        if all_correct:
            return 1.0
        if all_wrong:
            return -1.0

        correct = (target_is_good and vote_for_good) or ((not target_is_good) and (not vote_for_good))
        return 0.4 if correct else -0.4

    def choose_action(self, state: str):
        return self.learner.select_action(state)

    def decide_vote(
        self,
        voter: Node,
        target: Node,
        malicious_voter: bool,
        malicious_attack: bool,
    ) -> tuple[str, str, bool, bool]:
        """
        Returns:
            state, action, vote_for_good, explored
        """
        state = self.build_state(voter, target)
        sel = self.choose_action(state)

        if malicious_voter:
            vote_for_good = self.malicious_vote(
                target_is_bad=(target.state == NodeState.BAD),
                attack=malicious_attack,
            )
        else:
            vote_for_good = self.good_vote_by_history(voter, target, sel.action)

        return state, sel.action, vote_for_good, sel.explored

    def update_after_round(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
    ) -> None:
        self.learner.update(state, action, reward, next_state)
        self.learner.decay()