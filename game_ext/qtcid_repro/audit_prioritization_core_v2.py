"""
Переработанный модуль приоритизации системной проверки узлов.

КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ:
1. Убрана утечка ground truth из disagreement
2. Реализовано реальное локальное голосование с соседями
3. Добавлены типы узлов: benign_stable, benign_noisy, malicious_intermittent, malicious_persistent
4. Disagreement вычисляется из реальных локальных голосов
5. History-based method усилен EMA и stability tracking
6. Метрики исправлены: precision ≠ hit_rate
7. Добавлены новые метрики для шумных сценариев
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
import statistics
from typing import Dict, List, Set

from game_ext.qtcid_repro.wang.bvs_core import WangBVSConfig
from game_ext.qtcid_repro.mitchell.energy import (
    MitchellEnergyConfig,
    audit_energy,
    etids,
)
from game_ext.qtcid_repro.utils import binomial_sample, safe_div, clamp01


class NodeBehaviorType(Enum):
    """Типы поведения узлов"""
    BENIGN_STABLE = "benign_stable"        # Нормальный, низкий шум
    BENIGN_NOISY = "benign_noisy"          # Нормальный, но шумный
    MALICIOUS_INTERMITTENT = "malicious_intermittent"  # Вредоносный, эпизодический
    MALICIOUS_PERSISTENT = "malicious_persistent"      # Вредоносный, устойчивый


@dataclass
class NodeHistory:
    """История решений и наблюдений по узлу"""
    node_id: int
    behavior_type: NodeBehaviorType = NodeBehaviorType.BENIGN_STABLE

    # История локальных голосований (НЕТ прямого доступа к is_bad!)
    local_vote_history: List[List[int]] = field(default_factory=list)  # список голосов соседей за последние циклы
    disagreement_history: List[float] = field(default_factory=list)
    anomaly_count_history: List[int] = field(default_factory=list)

    # История расхождений между локальными решениями и аудитом
    mismatch_count: int = 0
    match_count: int = 0
    audit_count: int = 0

    # EMA (Exponential Moving Average) для mismatch
    ema_mismatch: float = 0.0
    ema_alpha: float = 0.3  # Скорость обновления

    # Стабильность
    stable_cycles: int = 0
    unstable_cycles: int = 0

    # Недавние аномалии (с decay)
    recent_anomalies: List[float] = field(default_factory=list)  # weighted by recency

    def stability_index(self) -> float:
        """Индекс стабильности на основе истории аудитов"""
        if self.audit_count == 0:
            return 0.5  # Нейтральное значение

        confirmation_ratio = safe_div(self.match_count, self.audit_count)
        time_factor = min(1.0, self.stable_cycles / 10.0)
        return clamp01(confirmation_ratio * (0.7 + 0.3 * time_factor))

    def mismatch_ratio(self) -> float:
        """Доля расхождений между локальными решениями и аудитом"""
        if self.audit_count == 0:
            return 0.0
        return safe_div(self.mismatch_count, self.audit_count)

    def recent_anomaly_score(self) -> float:
        """Взвешенная сумма недавних аномалий с экспоненциальным убыванием"""
        if not self.recent_anomalies:
            return 0.0
        # Более новые аномалии имеют больший вес
        weights = [0.75 ** i for i in range(len(self.recent_anomalies))]
        weighted_sum = sum(w * a for w, a in zip(weights, self.recent_anomalies))
        return clamp01(weighted_sum)


@dataclass
class AuditPrioritizationConfig(WangBVSConfig):
    """Конфигурация метода приоритизации системной проверки"""

    # Режим приоритизации
    prioritization_mode: str = "history_based"  # "random", "current_only", "history_based"

    # Бюджет аудита
    audit_budget_fixed: int = 5
    audit_budget_fraction: float = 0.0  # если > 0, используется вместо fixed

    # Коэффициенты priority score для history_based
    priority_current_disagreement: float = 0.30
    priority_current_anomaly: float = 0.20
    priority_ema_mismatch: float = 0.25
    priority_persistence: float = 0.15
    priority_stability_penalty: float = 0.35

    # Коэффициенты для current_only (без истории!)
    current_only_disagreement: float = 0.60
    current_only_anomaly: float = 0.40

    # Параметры локального голосования
    n_local_observers: int = 5  # число наблюдателей на узел
    observer_error_rate: float = 0.15  # вероятность ошибки наблюдателя

    # Параметры поведения узлов
    benign_noise_level: float = 0.2  # вероятность ложной тревоги для benign_noisy
    intermittent_activity_prob: float = 0.4  # вероятность активности malicious_intermittent
    persistent_activity_prob: float = 0.9  # вероятность активности malicious_persistent

    # Распределение типов узлов при компрометации
    prob_intermittent_on_capture: float = 0.6  # 60% новых bad узлов - intermittent
    prob_persistent_on_capture: float = 0.4    # 40% - persistent

    # Распределение типов benign узлов
    prob_benign_noisy: float = 0.3  # 30% benign узлов - noisy

    # Параметры истории
    history_window: int = 10
    ema_decay: float = 0.7  # для EMA mismatch

    # Порог для alarm
    alarm_threshold: float = 0.5  # порог для подсчёта anomaly


@dataclass
class AuditPrioritizationResult:
    """Результат одного прогона приоритизации"""
    mttf: float
    byzantine_failed: bool
    energy_failed: bool

    # Базовые метрики
    tp: int
    tn: int
    fp: int
    fn: int

    energy_left: float
    energy_spent_voting: float
    energy_spent_audit: float

    good_left: int
    bad_left: int
    evicted_left: int

    # Метрики приоритизации
    total_audits: int
    audits_on_bad: int
    audits_on_good: int

    # Новые метрики для разных типов
    audits_on_intermittent: int
    audits_on_persistent: int
    audits_on_benign_noisy: int
    audits_on_benign_stable: int

    total_intermittent: int
    total_persistent: int
    total_benign_noisy: int
    total_benign_stable: int

    cycles_to_verify_intermittent: List[int]
    cycles_to_verify_persistent: List[int]

    residual_risk_per_cycle: List[int]
    cumulative_residual_risk: float

    false_attention_events: int  # сколько раз benign_noisy попали в top-k


class AuditPrioritizationSimulator:
    """Симулятор приоритизации с реальным локальным голосованием"""

    def __init__(self, cfg: AuditPrioritizationConfig, seed: int) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)

        self.t = 0
        self.cycle = 0

        # Состояние системы
        self.ng = cfg.n_nodes
        self.nb = 0
        self.ne = 0

        self.energy_left = cfg.initial_system_energy

        # История по узлам
        self.node_histories: Dict[int, NodeHistory] = {}

        # Инициализация: распределить типы benign узлов
        for i in range(cfg.n_nodes):
            if self.rng.random() < cfg.prob_benign_noisy:
                behavior = NodeBehaviorType.BENIGN_NOISY
            else:
                behavior = NodeBehaviorType.BENIGN_STABLE
            self.node_histories[i] = NodeHistory(node_id=i, behavior_type=behavior)

        # Ground truth (для проверки в аудите, НЕ для вычисления scores!)
        self.bad_node_ids: Set[int] = set()
        self.evicted_node_ids: Set[int] = set()

        # Tracking компрометации
        self.compromise_cycle: Dict[int, int] = {}

        # Метрики
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.energy_spent_voting = 0.0
        self.energy_spent_audit = 0.0

        self.total_audits = 0
        self.audits_on_bad = 0
        self.audits_on_good = 0

        self.audits_on_intermittent = 0
        self.audits_on_persistent = 0
        self.audits_on_benign_noisy = 0
        self.audits_on_benign_stable = 0

        self.cycles_to_verify_intermittent: List[int] = []
        self.cycles_to_verify_persistent: List[int] = []

        self.residual_risk_per_cycle: List[int] = []
        self.false_attention_events = 0

        # Energy config
        self.energy_cfg = MitchellEnergyConfig(
            n_nodes=cfg.n_nodes,
            n_neighbors=cfg.n_neighbors,
            m_voters=cfg.m_voters,
            et=cfg.et,
            er=cfg.er,
            ea=cfg.ea,
            es=cfg.es,
            alpha_ranging=cfg.alpha_ranging,
        )

        self.vote_energy_value = etids(self.energy_cfg)
        self.audit_energy_value = audit_energy(self.energy_cfg, contacted_fraction=0.5)

    def active_node_ids(self) -> List[int]:
        """Список ID активных узлов"""
        return [nid for nid in range(self.cfg.n_nodes) if nid not in self.evicted_node_ids]

    def active_total(self) -> int:
        return len(self.active_node_ids())

    def byzantine_failure(self) -> bool:
        active = self.active_total()
        if active <= 0:
            return True
        return self.nb >= active / 3.0

    def energy_failure(self) -> bool:
        return self.energy_left <= 0.0

    # -------------------------------------------------------------------------
    # Capture process
    # -------------------------------------------------------------------------
    def capture_step(self) -> None:
        """Компрометация good узлов"""
        from math import exp

        good_ids = [nid for nid in self.active_node_ids() if nid not in self.bad_node_ids]
        if len(good_ids) == 0:
            return

        p_cap = 1.0 - exp(-self.cfg.lambda_capture * self.cfg.tids)
        new_bad_count = binomial_sample(self.rng, len(good_ids), p_cap)

        if new_bad_count > 0:
            captured = self.rng.sample(good_ids, new_bad_count)
            for nid in captured:
                self.bad_node_ids.add(nid)
                self.compromise_cycle[nid] = self.cycle

                # Присвоить тип malicious поведения
                if self.rng.random() < self.cfg.prob_intermittent_on_capture:
                    self.node_histories[nid].behavior_type = NodeBehaviorType.MALICIOUS_INTERMITTENT
                else:
                    self.node_histories[nid].behavior_type = NodeBehaviorType.MALICIOUS_PERSISTENT

                self.ng -= 1
                self.nb += 1

    # -------------------------------------------------------------------------
    # Реальное локальное голосование
    # -------------------------------------------------------------------------
    def conduct_local_voting(self) -> None:
        """
        Реальное локальное голосование: каждый узел оценивается соседями.

        Ключевое отличие: НЕТ прямого доступа к ground truth!
        Голоса формируются на основе:
        - типа поведения узла
        - ошибок наблюдателей
        - текущей активности узла
        """
        active = self.active_node_ids()
        if len(active) < self.cfg.n_local_observers:
            return

        # Энергия на голосование
        active_ratio = safe_div(len(active), self.cfg.n_nodes)
        interval_vote_energy = self.vote_energy_value * (0.35 + 0.65 * active_ratio)
        self.energy_left -= interval_vote_energy
        self.energy_spent_voting += interval_vote_energy

        for nid in active:
            hist = self.node_histories[nid]
            behavior = hist.behavior_type

            # Выбрать случайных наблюдателей (соседей)
            observers = self.rng.sample(active, min(self.cfg.n_local_observers, len(active)))

            # Каждый наблюдатель даёт оценку: 0 (good) или 1 (suspicious)
            votes = []
            for observer_id in observers:
                vote = self._observer_vote(nid, behavior)
                votes.append(vote)

            # Сохранить голоса в историю
            hist.local_vote_history.append(votes)
            if len(hist.local_vote_history) > self.cfg.history_window:
                hist.local_vote_history.pop(0)

            # Вычислить disagreement из голосов
            disagreement = self._compute_disagreement_from_votes(votes)
            hist.disagreement_history.append(disagreement)
            if len(hist.disagreement_history) > self.cfg.history_window:
                hist.disagreement_history.pop(0)

            # Подсчитать anomaly count
            suspicious_ratio = sum(votes) / len(votes)
            is_anomaly = 1 if suspicious_ratio >= self.cfg.alarm_threshold else 0
            hist.anomaly_count_history.append(is_anomaly)
            if len(hist.anomaly_count_history) > self.cfg.history_window:
                hist.anomaly_count_history.pop(0)

            # Обновить recent anomalies с decay
            if is_anomaly:
                hist.recent_anomalies.insert(0, 1.0)
            if len(hist.recent_anomalies) > 5:
                hist.recent_anomalies.pop()

    def _observer_vote(self, target_id: int, behavior: NodeBehaviorType) -> int:
        """
        Один наблюдатель оценивает целевой узел.

        Возвращает: 0 (считает good) или 1 (считает suspicious)

        НЕТ прямого доступа к is_bad! Голос зависит от:
        - типа поведения узла
        - ошибок наблюдателя
        - текущей активности узла
        """
        # Истинный сигнал от узла (зависит от типа поведения)
        if behavior == NodeBehaviorType.BENIGN_STABLE:
            true_signal = 0  # почти всегда корректный
            if self.rng.random() < 0.05:  # 5% шума
                true_signal = 1

        elif behavior == NodeBehaviorType.BENIGN_NOISY:
            true_signal = 0
            if self.rng.random() < self.cfg.benign_noise_level:  # шумный benign
                true_signal = 1

        elif behavior == NodeBehaviorType.MALICIOUS_INTERMITTENT:
            # Вредоносный, но не всегда проявляется
            if self.rng.random() < self.cfg.intermittent_activity_prob:
                true_signal = 1  # активен в этом цикле
            else:
                true_signal = 0  # неактивен

        elif behavior == NodeBehaviorType.MALICIOUS_PERSISTENT:
            # Вредоносный с устойчивым поведением
            if self.rng.random() < self.cfg.persistent_activity_prob:
                true_signal = 1
            else:
                true_signal = 0  # небольшая вероятность затаиться

        else:
            true_signal = 0

        # Применить ошибку наблюдателя
        if self.rng.random() < self.cfg.observer_error_rate:
            return 1 - true_signal  # перевернуть оценку
        else:
            return true_signal

    def _compute_disagreement_from_votes(self, votes: List[int]) -> float:
        """
        Вычислить disagreement из реальных голосов.

        Disagreement высокий, если голоса противоречивы (некоторые 0, некоторые 1).
        Disagreement низкий, если голоса согласованы (все 0 или все 1).
        """
        if len(votes) == 0:
            return 0.0

        suspicious_count = sum(votes)
        good_count = len(votes) - suspicious_count

        # Максимальное disagreement при 50/50 разделении
        ratio = suspicious_count / len(votes)

        # Entropy-based disagreement: максимум при 0.5, минимум при 0 и 1
        if ratio == 0.0 or ratio == 1.0:
            return 0.0  # Полное согласие
        else:
            # Entropy measure: -p*log(p) - (1-p)*log(1-p)
            import math
            entropy = -ratio * math.log2(ratio) - (1-ratio) * math.log2(1-ratio)
            return entropy  # range [0, 1]

    # -------------------------------------------------------------------------
    # Priority score computation
    # -------------------------------------------------------------------------
    def compute_current_only_score(self, nid: int) -> float:
        """
        Current-only score: только текущие наблюдения, БЕЗ истории.

        Использует:
        - текущий disagreement (последний цикл)
        - текущий anomaly count (последний цикл)

        НЕ использует:
        - EMA mismatch
        - stability index
        - persistence patterns
        """
        if nid in self.evicted_node_ids:
            return -1.0

        hist = self.node_histories[nid]

        # Текущий disagreement (последний)
        current_disagreement = hist.disagreement_history[-1] if hist.disagreement_history else 0.0

        # Текущий anomaly
        current_anomaly = float(hist.anomaly_count_history[-1]) if hist.anomaly_count_history else 0.0

        score = (
            self.cfg.current_only_disagreement * current_disagreement
            + self.cfg.current_only_anomaly * current_anomaly
        )

        return max(0.0, score)

    def compute_history_based_score(self, nid: int) -> float:
        """
        History-based priority score: текущие наблюдения + история.

        Использует:
        - текущий disagreement
        - текущий anomaly
        - EMA mismatch (память о расхождениях с аудитом)
        - persistence score (повторяющиеся аномалии)
        - stability penalty (длительно стабильные узлы)
        """
        if nid in self.evicted_node_ids:
            return -1.0

        hist = self.node_histories[nid]

        # Текущие признаки
        current_disagreement = hist.disagreement_history[-1] if hist.disagreement_history else 0.0
        current_anomaly = float(hist.anomaly_count_history[-1]) if hist.anomaly_count_history else 0.0

        # История
        ema_mismatch = hist.ema_mismatch
        persistence = self._compute_persistence_score(hist)
        stability = hist.stability_index()

        score = (
            self.cfg.priority_current_disagreement * current_disagreement
            + self.cfg.priority_current_anomaly * current_anomaly
            + self.cfg.priority_ema_mismatch * ema_mismatch
            + self.cfg.priority_persistence * persistence
            - self.cfg.priority_stability_penalty * stability
        )

        return max(0.0, score)

    def _compute_persistence_score(self, hist: NodeHistory) -> float:
        """
        Persistence score: насколько часто узел вызывает тревоги.

        Использует recent_anomaly_score + долю аномалий в истории.
        """
        recent_score = hist.recent_anomaly_score()

        if len(hist.anomaly_count_history) > 0:
            historical_ratio = sum(hist.anomaly_count_history) / len(hist.anomaly_count_history)
        else:
            historical_ratio = 0.0

        return 0.6 * recent_score + 0.4 * historical_ratio

    # -------------------------------------------------------------------------
    # Node selection for audit
    # -------------------------------------------------------------------------
    def select_nodes_for_audit(self) -> List[int]:
        """Выбор узлов для системного аудита с учётом бюджета"""
        active = self.active_node_ids()
        if len(active) == 0:
            return []

        # Определить размер бюджета
        if self.cfg.audit_budget_fraction > 0:
            budget = max(1, int(self.cfg.audit_budget_fraction * len(active)))
        else:
            budget = min(self.cfg.audit_budget_fixed, len(active))

        if self.cfg.prioritization_mode == "random":
            return self.rng.sample(active, budget)

        elif self.cfg.prioritization_mode == "current_only":
            scores = [(nid, self.compute_current_only_score(nid)) for nid in active]
            scores.sort(key=lambda x: x[1], reverse=True)
            return [nid for nid, _ in scores[:budget]]

        elif self.cfg.prioritization_mode == "history_based":
            scores = [(nid, self.compute_history_based_score(nid)) for nid in active]
            scores.sort(key=lambda x: x[1], reverse=True)
            return [nid for nid, _ in scores[:budget]]

        else:
            raise ValueError(f"Unknown prioritization_mode: {self.cfg.prioritization_mode}")

    # -------------------------------------------------------------------------
    # Audit step
    # -------------------------------------------------------------------------
    def audit_step(self) -> None:
        """Системная проверка выбранных узлов"""
        selected_nodes = self.select_nodes_for_audit()

        if len(selected_nodes) == 0:
            return

        # Энергия на аудит
        audit_energy_cost = len(selected_nodes) * self.audit_energy_value
        self.energy_left -= audit_energy_cost
        self.energy_spent_audit += audit_energy_cost

        for nid in selected_nodes:
            self.total_audits += 1
            hist = self.node_histories[nid]
            hist.audit_count += 1

            is_bad = nid in self.bad_node_ids  # Ground truth

            # Обновить метрики по типам
            if hist.behavior_type == NodeBehaviorType.BENIGN_STABLE:
                self.audits_on_benign_stable += 1
            elif hist.behavior_type == NodeBehaviorType.BENIGN_NOISY:
                self.audits_on_benign_noisy += 1
                self.false_attention_events += 1  # ложное внимание к шумному benign
            elif hist.behavior_type == NodeBehaviorType.MALICIOUS_INTERMITTENT:
                self.audits_on_intermittent += 1
            elif hist.behavior_type == NodeBehaviorType.MALICIOUS_PERSISTENT:
                self.audits_on_persistent += 1

            if is_bad:
                # True Positive
                self.tp += 1
                self.audits_on_bad += 1

                self.bad_node_ids.remove(nid)
                self.evicted_node_ids.add(nid)
                self.nb -= 1
                self.ne += 1

                # Записать cycles to verify
                compromise_cycle = self.compromise_cycle.get(nid, self.cycle)
                cycles_elapsed = self.cycle - compromise_cycle

                if hist.behavior_type == NodeBehaviorType.MALICIOUS_INTERMITTENT:
                    self.cycles_to_verify_intermittent.append(cycles_elapsed)
                elif hist.behavior_type == NodeBehaviorType.MALICIOUS_PERSISTENT:
                    self.cycles_to_verify_persistent.append(cycles_elapsed)

                # Обновить историю: match (локальная тревога подтвердилась)
                hist.match_count += 1
                hist.stable_cycles = 0

                # Обновить EMA mismatch (при match EMA снижается)
                hist.ema_mismatch = hist.ema_mismatch * (1 - hist.ema_alpha)

            else:
                # Good узел
                self.audits_on_good += 1

                # Проверить: было ли ложное обвинение
                recent_votes = hist.local_vote_history[-1] if hist.local_vote_history else []
                suspicious_ratio = sum(recent_votes) / len(recent_votes) if recent_votes else 0

                if suspicious_ratio >= self.cfg.alarm_threshold:
                    # False Positive: был alarm, но узел good
                    self.fp += 1
                    hist.mismatch_count += 1

                    # Обновить EMA mismatch (при mismatch EMA растёт)
                    hist.ema_mismatch = hist.ema_mismatch * (1 - hist.ema_alpha) + hist.ema_alpha

                    hist.stable_cycles = 0
                    hist.unstable_cycles += 1
                else:
                    # True Negative: правильно определён как good
                    self.tn += 1
                    hist.match_count += 1

                    # Обновить EMA mismatch (при match EMA снижается)
                    hist.ema_mismatch = hist.ema_mismatch * (1 - hist.ema_alpha)

                    hist.stable_cycles += 1

    def update_residual_risk(self) -> None:
        """Обновить остаточный риск: число bad узлов, не проверенных аудитом"""
        unverified_bad = len([nid for nid in self.bad_node_ids if nid not in self.evicted_node_ids])
        self.residual_risk_per_cycle.append(unverified_bad)

    # -------------------------------------------------------------------------
    # Main simulation loop
    # -------------------------------------------------------------------------
    def run(self) -> AuditPrioritizationResult:
        """Основной цикл симуляции"""
        while self.t <= self.cfg.max_time:
            if self.byzantine_failure() or self.energy_failure():
                break

            self.cycle += 1

            # Компрометация узлов
            self.capture_step()

            if self.byzantine_failure() or self.energy_failure():
                break

            # Локальное голосование (БЕЗ доступа к ground truth!)
            self.conduct_local_voting()

            # Системный аудит с приоритизацией
            self.audit_step()

            # Обновить residual risk
            self.update_residual_risk()

            if self.byzantine_failure() or self.energy_failure():
                break

            self.t += self.cfg.tids

        # False Negatives: bad узлы, которые остались
        unverified_bad = len([nid for nid in self.bad_node_ids if nid not in self.evicted_node_ids])
        self.fn = unverified_bad

        cumulative_residual_risk = sum(self.residual_risk_per_cycle)

        # Подсчитать типы узлов
        total_intermittent = sum(1 for h in self.node_histories.values() if h.behavior_type == NodeBehaviorType.MALICIOUS_INTERMITTENT and h.node_id not in self.evicted_node_ids)
        total_persistent = sum(1 for h in self.node_histories.values() if h.behavior_type == NodeBehaviorType.MALICIOUS_PERSISTENT and h.node_id not in self.evicted_node_ids)
        total_benign_noisy = sum(1 for h in self.node_histories.values() if h.behavior_type == NodeBehaviorType.BENIGN_NOISY and h.node_id not in self.evicted_node_ids)
        total_benign_stable = sum(1 for h in self.node_histories.values() if h.behavior_type == NodeBehaviorType.BENIGN_STABLE and h.node_id not in self.evicted_node_ids)

        return AuditPrioritizationResult(
            mttf=float(self.t),
            byzantine_failed=self.byzantine_failure(),
            energy_failed=self.energy_failure(),
            tp=self.tp,
            tn=self.tn,
            fp=self.fp,
            fn=self.fn,
            energy_left=self.energy_left,
            energy_spent_voting=self.energy_spent_voting,
            energy_spent_audit=self.energy_spent_audit,
            good_left=self.ng,
            bad_left=self.nb,
            evicted_left=self.ne,
            total_audits=self.total_audits,
            audits_on_bad=self.audits_on_bad,
            audits_on_good=self.audits_on_good,
            audits_on_intermittent=self.audits_on_intermittent,
            audits_on_persistent=self.audits_on_persistent,
            audits_on_benign_noisy=self.audits_on_benign_noisy,
            audits_on_benign_stable=self.audits_on_benign_stable,
            total_intermittent=total_intermittent,
            total_persistent=total_persistent,
            total_benign_noisy=total_benign_noisy,
            total_benign_stable=total_benign_stable,
            cycles_to_verify_intermittent=self.cycles_to_verify_intermittent,
            cycles_to_verify_persistent=self.cycles_to_verify_persistent,
            residual_risk_per_cycle=self.residual_risk_per_cycle,
            cumulative_residual_risk=cumulative_residual_risk,
            false_attention_events=self.false_attention_events,
        )


def summarize_prioritization(results: List[AuditPrioritizationResult]) -> dict:
    """Агрегирование результатов множества прогонов"""
    def mean(values):
        return statistics.mean(values) if values else 0.0

    def std(values):
        return statistics.pstdev(values) if len(values) > 1 else 0.0

    # Базовые метрики
    mttf_vals = [r.mttf for r in results]
    tp_vals = [r.tp for r in results]
    tn_vals = [r.tn for r in results]
    fp_vals = [r.fp for r in results]
    fn_vals = [r.fn for r in results]

    tp_mean = mean(tp_vals)
    tn_mean = mean(tn_vals)
    fp_mean = mean(fp_vals)
    fn_mean = mean(fn_vals)

    total = tp_mean + tn_mean + fp_mean + fn_mean
    accuracy = safe_div(tp_mean + tn_mean, total)

    # Метрики приоритизации
    total_audits = mean([r.total_audits for r in results])
    audits_on_bad = mean([r.audits_on_bad for r in results])
    audits_on_good = mean([r.audits_on_good for r in results])

    # ИСПРАВЛЕНО: precision и recall - разные метрики!
    audit_precision = safe_div(audits_on_bad, total_audits)  # доля bad среди выбранных

    # Recall (hit rate): доля bad, попавших в аудит
    total_bad = mean([r.tp + r.fn for r in results])  # всего bad узлов
    recall_hit_rate = safe_div(audits_on_bad, total_bad) if total_bad > 0 else 0.0

    # Метрики по типам
    audits_on_intermittent = mean([r.audits_on_intermittent for r in results])
    audits_on_persistent = mean([r.audits_on_persistent for r in results])
    audits_on_benign_noisy = mean([r.audits_on_benign_noisy for r in results])
    audits_on_benign_stable = mean([r.audits_on_benign_stable for r in results])

    total_intermittent = mean([r.total_intermittent for r in results])
    total_persistent = mean([r.total_persistent for r in results])
    total_benign_noisy = mean([r.total_benign_noisy for r in results])
    total_benign_stable = mean([r.total_benign_stable for r in results])

    intermittent_detection_rate = safe_div(audits_on_intermittent, total_intermittent + audits_on_intermittent) if (total_intermittent + audits_on_intermittent) > 0 else 0.0
    persistent_detection_rate = safe_div(audits_on_persistent, total_persistent + audits_on_persistent) if (total_persistent + audits_on_persistent) > 0 else 0.0

    false_attention_rate = mean([r.false_attention_events /max(1, r.total_audits) for r in results])

    # Cycles to verify
    all_cycles_intermittent = []
    all_cycles_persistent = []
    for r in results:
        all_cycles_intermittent.extend(r.cycles_to_verify_intermittent)
        all_cycles_persistent.extend(r.cycles_to_verify_persistent)

    mean_cycles_intermittent = mean(all_cycles_intermittent) if all_cycles_intermittent else 0.0
    mean_cycles_persistent = mean(all_cycles_persistent) if all_cycles_persistent else 0.0

    # Cumulative residual risk
    cumulative_residual_risk = mean([r.cumulative_residual_risk for r in results])

    return {
        "runs": len(results),
        "mttf_mean": mean(mttf_vals),
        "mttf_std": std(mttf_vals),
        "accuracy_mean": accuracy,
        "tp_mean": tp_mean,
        "tn_mean": tn_mean,
        "fp_mean": fp_mean,
        "fn_mean": fn_mean,
        "total_audits_mean": total_audits,
        "audits_on_bad_mean": audits_on_bad,
        "audits_on_good_mean": audits_on_good,
        "audit_precision_mean": audit_precision,
        "recall_hit_rate_mean": recall_hit_rate,
        "audits_on_intermittent_mean": audits_on_intermittent,
        "audits_on_persistent_mean": audits_on_persistent,
        "audits_on_benign_noisy_mean": audits_on_benign_noisy,
        "audits_on_benign_stable_mean": audits_on_benign_stable,
        "intermittent_detection_rate_mean": intermittent_detection_rate,
        "persistent_detection_rate_mean": persistent_detection_rate,
        "false_attention_rate_mean": false_attention_rate,
        "mean_cycles_to_verify_intermittent": mean_cycles_intermittent,
        "mean_cycles_to_verify_persistent": mean_cycles_persistent,
        "cumulative_residual_risk_mean": cumulative_residual_risk,
        "cumulative_residual_risk_std": std([r.cumulative_residual_risk for r in results]),
    }


def run_prioritization_monte_carlo(cfg: AuditPrioritizationConfig) -> dict:
    """Прогон Монте-Карло"""
    results: List[AuditPrioritizationResult] = []
    for i in range(cfg.runs):
        sim = AuditPrioritizationSimulator(cfg, seed=cfg.seed + i)
        results.append(sim.run())
    return summarize_prioritization(results)
