"""
Модуль приоритизации системной проверки узлов на основе истории согласованности решений.

Задача: при ограниченном бюджете аудита выбрать наиболее подозрительные узлы
на основе истории локальных коллективных оценок.

Отличие от IDS: выходом является приоритет проверки и выбор top-k узлов,
а не финальное решение о состоянии узла.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import statistics
from typing import Dict, List

from game_ext.qtcid_repro.wang.bvs_core import WangBVSConfig
from game_ext.qtcid_repro.mitchell.energy import (
    MitchellEnergyConfig,
    audit_energy,
    etids,
)
from game_ext.qtcid_repro.mitchell.voting import wang_ids_error_probability
from game_ext.qtcid_repro.utils import binomial_sample, safe_div, clamp01


@dataclass
class NodeHistory:
    """История решений и наблюдений по узлу"""
    node_id: int
    is_bad: bool = False  # Ground truth (известен симулятору, но не системе)

    # История расхождений в локальных оценках
    disagreement_events: List[float] = field(default_factory=list)

    # История расхождений между локальными решениями и аудитом
    mismatch_count: int = 0
    audit_count: int = 0

    # История подтверждённых проблем
    confirmed_suspicious: int = 0
    confirmed_correct: int = 0

    # Недавние аномалии
    recent_suspicions: List[int] = field(default_factory=list)  # last N cycles

    # Стабильность
    stable_cycles: int = 0

    # Связи с проблемными узлами
    suspicious_neighbor_count: int = 0
    total_neighbor_count: int = 0

    def stability_score(self) -> float:
        """Оценка стабильности поведения узла"""
        total_audits = max(1, self.audit_count)
        confirmation_ratio = safe_div(self.confirmed_correct, total_audits)
        stability = confirmation_ratio * min(1.0, self.stable_cycles / 10.0)
        return clamp01(stability)

    def mismatch_ratio(self) -> float:
        """Доля расхождений между локальными решениями и аудитом"""
        if self.audit_count == 0:
            return 0.0
        return safe_div(self.mismatch_count, self.audit_count)

    def suspicious_neighbor_ratio(self) -> float:
        """Доля подозрительных соседей"""
        if self.total_neighbor_count == 0:
            return 0.0
        return safe_div(self.suspicious_neighbor_count, self.total_neighbor_count)

    def recent_anomaly_score(self) -> float:
        """Оценка недавних аномалий"""
        if not self.recent_suspicions:
            return 0.0
        # Взвешенная сумма с убыванием для старых событий
        weights = [0.8 ** i for i in range(len(self.recent_suspicions))]
        weighted_sum = sum(w * s for w, s in zip(weights, self.recent_suspicions))
        return clamp01(weighted_sum)


@dataclass
class AuditPrioritizationConfig(WangBVSConfig):
    """
    Конфигурация метода приоритизации системной проверки.

    Ключевые отличия от IDS:
    - audit_budget: ограниченный бюджет проверок (число узлов или доля)
    - prioritization_mode: метод выбора узлов для аудита
    - priority_coefficients: веса компонентов priority score
    """

    # Режим приоритизации
    prioritization_mode: str = "history_based"  # "random", "current_only", "history_based"

    # Бюджет аудита (число узлов на цикл или доля от активных)
    audit_budget_fixed: int = 5  # фиксированное число узлов
    audit_budget_fraction: float = 0.0  # доля активных узлов (если > 0, используется вместо fixed)

    # Коэффициенты priority score
    priority_disagreement_weight: float = 0.35
    priority_mismatch_weight: float = 0.30
    priority_neighbor_weight: float = 0.15
    priority_anomaly_weight: float = 0.20
    priority_stability_penalty: float = 0.25

    # История: сколько циклов помнить
    history_window: int = 10

    # Probability для оценки disagreement (если нет реальных voters)
    simulated_disagreement_pa: float = 0.5


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

    # НОВЫЕ метрики приоритизации
    total_audits: int
    audits_on_bad: int  # hit count
    audits_on_good: int  # wasted audits

    cycles_to_verify_bad: List[int]  # cycles from compromise to audit for each bad node

    residual_risk_per_cycle: List[int]  # unaudited bad nodes per cycle
    cumulative_residual_risk: float  # sum of residual_risk_per_cycle


class AuditPrioritizationSimulator:
    """
    Симулятор приоритизации системных проверок с ограниченным бюджетом.

    Основные отличия от IDS:
    1. Не все узлы проверяются аудитом - только top-k
    2. Приоритет вычисляется на основе истории согласованности
    3. Метрики не accuracy, а hit_rate, residual_risk, precision
    """

    def __init__(self, cfg: AuditPrioritizationConfig, seed: int) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)

        self.t = 0
        self.cycle = 0

        # Состояние системы
        self.ng = cfg.n_nodes  # good nodes
        self.nb = 0  # bad nodes
        self.ne = 0  # evicted nodes

        self.energy_left = cfg.initial_system_energy

        # История по узлам
        self.node_histories: Dict[int, NodeHistory] = {}
        for i in range(cfg.n_nodes):
            self.node_histories[i] = NodeHistory(node_id=i, is_bad=False)

        # Текущие bad узлы (для ground truth)
        self.bad_node_ids: set[int] = set()
        self.evicted_node_ids: set[int] = set()

        # Tracking: когда узел стал bad
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

        self.cycles_to_verify_bad: List[int] = []
        self.residual_risk_per_cycle: List[int] = []

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
                self.node_histories[nid].is_bad = True
                self.compromise_cycle[nid] = self.cycle
                self.ng -= 1
                self.nb += 1

    # -------------------------------------------------------------------------
    # Priority score computation
    # -------------------------------------------------------------------------
    def compute_disagreement_index(self, nid: int) -> float:
        """
        Индекс расхождения локальных оценок по узлу.

        В упрощённой модели без явных voters используем симуляцию:
        расхождение выше, если среди соседей есть bad nodes.
        """
        hist = self.node_histories[nid]
        if not hist.disagreement_events:
            # Базовая оценка на основе current suspicion
            return 0.3 if hist.is_bad else 0.1
        return min(1.0, statistics.mean(hist.disagreement_events[-self.cfg.history_window:]))

    def compute_priority_score(self, nid: int) -> float:
        """
        Вычисление priority score для узла.

        Priority(i, t) = a1 * disagreement_index
                       + a2 * historical_mismatch
                       + a3 * suspicious_neighbor_ratio
                       + a4 * recent_anomaly
                       - a5 * stability_score
        """
        if nid in self.evicted_node_ids:
            return -1.0  # Excluded from audit

        hist = self.node_histories[nid]

        disagreement = self.compute_disagreement_index(nid)
        mismatch = hist.mismatch_ratio()
        neighbor_ratio = hist.suspicious_neighbor_ratio()
        anomaly = hist.recent_anomaly_score()
        stability = hist.stability_score()

        score = (
            self.cfg.priority_disagreement_weight * disagreement
            + self.cfg.priority_mismatch_weight * mismatch
            + self.cfg.priority_neighbor_weight * neighbor_ratio
            + self.cfg.priority_anomaly_weight * anomaly
            - self.cfg.priority_stability_penalty * stability
        )

        return max(0.0, score)

    # -------------------------------------------------------------------------
    # Node selection for audit
    # -------------------------------------------------------------------------
    def select_nodes_for_audit(self) -> List[int]:
        """
        Выбор узлов для системного аудита с учётом бюджета.

        Режимы:
        - random: случайный выбор
        - current_only: по текущей подозрительности (без истории)
        - history_based: по полному priority score
        """
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
            # Выбор по текущей подозрительности (только disagreement_index)
            scores = [(nid, self.compute_disagreement_index(nid)) for nid in active]
            scores.sort(key=lambda x: x[1], reverse=True)
            return [nid for nid, _ in scores[:budget]]

        elif self.cfg.prioritization_mode == "history_based":
            # Полный priority score
            scores = [(nid, self.compute_priority_score(nid)) for nid in active]
            scores.sort(key=lambda x: x[1], reverse=True)
            return [nid for nid, _ in scores[:budget]]

        else:
            raise ValueError(f"Unknown prioritization_mode: {self.cfg.prioritization_mode}")

    # -------------------------------------------------------------------------
    # Voting and audit
    # -------------------------------------------------------------------------
    def voting_step(self) -> None:
        """
        Локальное коллективное голосование (без immediate eviction).

        Результаты голосования обновляют disagreement_index и историю узлов,
        но финальное решение принимается только после аудита.
        """
        active = self.active_node_ids()
        if len(active) < self.cfg.m_voters:
            return

        # Энергия на голосование
        active_ratio = safe_div(len(active), self.cfg.n_nodes)
        interval_vote_energy = self.vote_energy_value * (0.35 + 0.65 * active_ratio)
        self.energy_left -= interval_vote_energy
        self.energy_spent_voting += interval_vote_energy

        # Симуляция disagreement events для всех узлов
        # В упрощённой модели: для bad узлов disagreement выше
        p_fp_ids = wang_ids_error_probability(
            n_good=self.ng,
            n_bad=self.nb,
            m=self.cfg.m_voters,
            pa=self.cfg.simulated_disagreement_pa,
            omega=self.cfg.hpfp,
        )

        for nid in active:
            hist = self.node_histories[nid]
            if hist.is_bad:
                # Bad узел: высокое disagreement
                disagreement_val = 0.5 + 0.3 * self.rng.random()
            else:
                # Good узел: низкое disagreement с небольшой вероятностью ошибки
                if self.rng.random() < p_fp_ids:
                    disagreement_val = 0.3 + 0.2 * self.rng.random()
                else:
                    disagreement_val = 0.05 + 0.1 * self.rng.random()

            hist.disagreement_events.append(disagreement_val)
            if len(hist.disagreement_events) > self.cfg.history_window:
                hist.disagreement_events.pop(0)

    def audit_step(self) -> None:
        """
        Системная проверка выбранных узлов с ограниченным бюджетом.

        Только выбранные узлы проходят аудит.
        Аудит определяет ground truth и обновляет метрики.
        """
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

            if hist.is_bad:
                # Узел действительно bad
                self.audits_on_bad += 1

                # True Positive
                self.tp += 1
                self.bad_node_ids.remove(nid)
                self.evicted_node_ids.add(nid)
                hist.confirmed_suspicious += 1
                self.nb -= 1
                self.ne += 1

                # Записать cycles to verify
                compromise_cycle = self.compromise_cycle.get(nid, self.cycle)
                cycles_elapsed = self.cycle - compromise_cycle
                self.cycles_to_verify_bad.append(cycles_elapsed)

            else:
                # Узел good
                self.audits_on_good += 1

                # Проверить: было ли ложное обвинение
                if hist.disagreement_events and hist.disagreement_events[-1] > 0.3:
                    # False Positive: узел был подозрителен, но оказался good
                    self.fp += 1
                    hist.mismatch_count += 1
                else:
                    # True Negative: правильно определён как good
                    self.tn += 1
                    hist.confirmed_correct += 1
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

            # Локальное голосование (обновление disagreement)
            self.voting_step()

            # Системный аудит с приоритизацией
            self.audit_step()

            # Обновить residual risk
            self.update_residual_risk()

            if self.byzantine_failure() or self.energy_failure():
                break

            self.t += self.cfg.tids

        # False Negatives: bad узлы, которые остались в системе
        unverified_bad = len([nid for nid in self.bad_node_ids if nid not in self.evicted_node_ids])
        self.fn = unverified_bad

        cumulative_residual_risk = sum(self.residual_risk_per_cycle)

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
            cycles_to_verify_bad=self.cycles_to_verify_bad,
            residual_risk_per_cycle=self.residual_risk_per_cycle,
            cumulative_residual_risk=cumulative_residual_risk,
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

    # НОВЫЕ метрики приоритизации
    total_audits = mean([r.total_audits for r in results])
    audits_on_bad = mean([r.audits_on_bad for r in results])
    audits_on_good = mean([r.audits_on_good for r in results])

    hit_rate = safe_div(audits_on_bad, total_audits)
    wasted_rate = safe_div(audits_on_good, total_audits)
    precision = hit_rate

    # Mean cycles to verify bad nodes
    all_cycles = []
    for r in results:
        all_cycles.extend(r.cycles_to_verify_bad)
    mean_cycles_to_verify = mean(all_cycles) if all_cycles else 0.0

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
        "hit_rate_mean": hit_rate,
        "wasted_rate_mean": wasted_rate,
        "precision_mean": precision,
        "mean_cycles_to_verify": mean_cycles_to_verify,
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
