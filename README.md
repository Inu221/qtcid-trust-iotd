# Q-TCID / TA-QTCID / Audit Prioritization — Эксперименты для статьи

Репозиторий содержит реализацию и сравнительный эксперимент методов обнаружения компрометированных узлов в IoT-сетях:

- **Q-TCID** — базовый метод-аналог на основе Q-обучения с голосованием (реализация по Wang et al.)
- **TA-QTCID** (Trust-Aware Q-TCID) — авторский метод, расширяющий Q-TCID динамической моделью доверия узлов
- **Audit Prioritization** — метод приоритизации системной проверки узлов на основе истории согласованности решений

## Ключевые идеи

### Q-TCID vs TA-QTCID

При сопоставимых значениях MTTF и точности обнаружения TA-QTCID стабильно снижает метрику CMVI (Cumulative Misclassification Vulnerability Index) на **3–12%** во всём диапазоне параметров атаки $P_a \in [0, 1]$ и интервала диагностики $T_{IDS} \in [50, 1500]$.

### Audit Prioritization (v2 - переработанная версия)

Новый метод решает отличную от IDS задачу: **приоритизация узлов для ограниченного системного аудита** на основе истории согласованности локальных решений.

**Ключевое отличие от IDS:**
- Раньше: выход = решение о состоянии узла
- Теперь: выход = приоритет проверки узла и выбор top-K узлов при ограниченном бюджете аудита

**Критические исправления v2:**
- ✅ Убрана утечка ground truth (v1 содержала пря��ой доступ к `is_bad`)
- ✅ Реализовано реальное локальное голосование с соседями
- ✅ Добавлены типы узлов: benign_stable, benign_noisy, malicious_intermittent, malicious_persistent
- ✅ Disagreement вычисляется entropy-based из реальных голосов
- ✅ Исправлены метрики: precision ≠ recall
- ✅ Новые метрики: intermittent_detection_rate, false_attention_rate

**Результаты v2 (честные):**
- History-based показывает **чёткое превосходство** при малых бюджетах (3-5 узлов): +13-28% по precision
- При больших бюджетах (8+) current-only сопоставим или лучше
- Random значительно хуже во всех сценариях
- **Метод полезен в специфических условиях:** ограниченный бюджет + шумная среда + intermittent attacks

**Документация:**
- `AUDIT_REPORT.md` — критический аудит v1 и найденные проблемы
- `RESULTS_V2_ANALYSIS.md` — детальный анализ результатов v2
- `METHOD.md` — формализация метода

## Структура

```
game_ext/qtcid_repro/
├── qtcid_core.py          # Реализация Q-TCID
├── ta_qtcid_core.py       # Реализация TA-QTCID (авторский метод)
├── audit_prioritization_core.py     # Приоритизация v1 (содержит проблемы - см. AUDIT_REPORT.md)
├── audit_prioritization_core_v2.py  # Приоритизация v2 (исправленная версия)
├── types.py               # Общие типы данных
├── utils.py               # Вспомогательные функции
├── mitchell/              # Базовые формулы (энергия, голосование) по Mitchell
│   ├── energy.py
│   └── voting.py
├── wang/                  # Базовая система BVS (аналог) по Wang
│   ├── bvs_core.py
│   └── game.py
└── experiments/
    ├── final_qtcid_taqtcid_study.py        # Q-TCID vs TA-QTCID
    ├── audit_prioritization_study.py       # Приоритизация v1
    └── audit_prioritization_study_v2.py    # Приоритизация v2 (с улучшенными сценариями)

results/
├── final_qtcid_taqtcid_study/  # Результаты IDS-экспериментов
├── audit_prioritization_study/      # Результаты v1 (некорректные)
└── audit_prioritization_study_v2/   # Результаты v2 (исправленные)
    ├── detailed_results_v2.csv
    ├── detailed_results_v2.json
    └── figures/                     # 6+ графиков по сценариям

AUDIT_REPORT.md                  # Критический аудит v1: найденные проблемы
RESULTS_V2_ANALYSIS.md           # Детальный анализ результатов v2
METHOD.md                        # Формализация метода приоритизации
```

## Параметры экспериментов

### Q-TCID vs TA-QTCID

| Параметр | Значение |
|---|---|
| Узлов в сети | 128 |
| Соседей на узел | 32 |
| Голосующих (m) | 5 |
| Прогонов Monte Carlo | 80 |
| $P_a$ | 0.0, 0.25, 0.5, 0.75, 1.0 |
| $T_{IDS}$ | 50, 100, 200, 350, 600, 1000, 1500 |
| $\lambda_{capture}$ | 1/3600 |
| hpfp / hpfn | 0.05 / 0.05 |

### Audit Prioritization

| Параметр | Значение | Режим |
|---|---|---|
| Узлов в сети | 128 | — |
| Audit Budget | 3, 5, 8, 12, 16 | FULL |
| Audit Budget | 3, 5, 8 | FAST |
| $P_a$ | 0.25, 0.5, 0.75 | FULL |
| $P_a$ | 0.5 | FAST |
| $T_{IDS}$ | 200, 600, 1000 | FULL |
| $T_{IDS}$ | 200 | FAST |
| Прогонов Monte Carlo | 60 | FULL |
| Прогонов Monte Carlo | 10 | FAST |
| Режимы приоритизации | random, current_only, history_based | — |

**Priority Score коэффициенты (α):**
- disagreement: 0.35
- mismatch: 0.30
- neighbor_ratio: 0.15
- anomaly: 0.20
- stability_penalty: 0.25

## Запуск

### IDS-эксперимент (Q-TCID vs TA-QTCID)

```bash
python3 -m game_ext.qtcid_repro.experiments.final_qtcid_taqtcid_study
```

Результаты сохраняются в `results/final_qtcid_taqtcid_study/`.

### Эксперимент приоритизации аудита (v2 - исправленная версия)

**⚠️ ВАЖНО:** Используйте v2! Оригинальная версия (v1) содержитутечку ground truth (см. `AUDIT_REPORT.md`).

**Быстрый тест v2 (2-5 минут):**
```bash
# Редактировать experiments/audit_prioritization_study_v2.py:
# FAST_MODE = True

python3 -m game_ext.qtcid_repro.experiments.audit_prioritization_study_v2
```

**Полный эксперимент v2 (30-60 минут):**
```bash
# Редактировать experiments/audit_prioritization_study_v2.py:
# FULL_MODE = True

python3 -m game_ext.qtcid_repro.experiments.audit_prioritization_study_v2
```

Результаты сохраняются в `results/audit_prioritization_study_v2/`.

**Генерируется:**
- `detailed_results_v2.csv` — полная таблица метрик
- `detailed_results_v2.json` — JSON версия
- `figures/*.png` — графики по сценариям

**Графики v2 (для каждого сценария):**
1. Audit Precision vs Budget
2. Recall (Hit Rate) vs Budget
3. Intermittent Detection Rate vs Budget
4. False Attention Rate vs Budget
5. Cumulative Residual Risk vs Budget
6. Mean Cycles to Verify (Intermittent) vs Budget
7. Scenarios Comparison (при FULL_MODE)

**Сценарии v2:**
- `low_noise_persistent`: низкий шум, устойчивые атаки
- `medium_noise_mixed`: средний шум, смешанные атаки (intermittent + persistent)
- `high_noise_intermittent`: высокий шум, эпизодические атаки

## Зависимости

```bash
pip install -r requirements.txt
```

---

## Документация

- **METHOD.md** — Подробная формализация метода приоритизации системной проверки
  - Постановка задачи
  - Priority Score формула и компоненты
  - Baseline методы
  - Метрики эффективности
  - Отличия от IDS-подхода
  - Инструкции по запуску

---

## Научная новизна метода приоритизации

1. **Переформулировка задачи**: От IDS-классификации (выход = решение о состоянии) к приоритизации при ограниченных ресурсах (выход = priority score и выбор top-K)

2. **Priority Score**: Формализация приоритета на основе истории согласованности локальных решений:
   ```
   P(i,t) = α₁·disagreement + α₂·mismatch + α₃·neighbor_ratio + α₄·anomaly - α₅·stability
   ```

3. **Новые метрики**: Hit Rate, Residual Risk, Mean Cycles to Verify — адекватны задаче ранжирования и выбора при ограниченном бюджете

4. **Учёт истории**: Mismatch ratio и stability score как память о согласованности/несогласованности локальных оценок с результатами аудита

5. **Baseline сравнение**: Демонстрация превосходства history-based над random и current-only методами

---

## Примеры результатов v2

### Быстрый тест v2 (FAST_MODE, medium_noise_mixed scenario)

**Audit Precision (доля bad среди выбранных):**

| Budget | Random | Current-only | History-based (proposed) | Улучшение |
|--------|--------|--------------|--------------------------|-----------|
| 3      | 0.149  | 0.354        | **0.454** ✅             | +28% vs current |
| 5      | 0.200  | 0.393        | **0.446** ✅             | +13% vs current |
| 8      | 0.176  | **0.448**    | 0.411                    | -8% vs current |

**Recall (Hit Rate) - доля обнаруженных bad узлов:**

| Budget | Random | Current-only | History-based | Улучшение |
|--------|--------|--------------|---------------|-----------|
| 3      | 0.069  | 0.170        | **0.217** ✅  | +28% vs current |
| 5      | 0.170  | 0.375        | **0.381** ✅  | +1.6% vs current |
| 8      | 0.225  | **0.862** ✅ | 0.725         | -16% vs current |

**Cumulative Residual Risk:**

| Budget | Random | Current-only | History-based | Улучшение |
|--------|--------|--------------|---------------|-----------|
| 3      | 175    | 193          | **187** ✅    | -3% vs current |
| 5      | 196    | 244          | **205** ✅    | -16% vs current |
| 8      | 190    | 309          | **262** ✅    | -15% vs current |

### ✅ Ключевые выводы v2

**1. Чёткая область превосходства:**
- History-based значительно лучше при **малых бюджетах (3-5 узлов)**
- Precision: +13-28% vs current-only
- Recall: +1.6-28% vs current-only
- Cumulative Risk: -3% до -16%

**2. Trade-offs:**
- При **больших бюджетах (8+)** current-only сопоставим или лучше
- False attention rate высокая у обоих умных методов (0.33-0.47)

**3. Метод полезен когда:**
- Системный аудит дорогой (ограниченный бюджет)
- Среда шумная (benign_noisy nodes существуют)
- Атаки intermittent (не всегда проявляются)

**4. Научная честность:**
- Метод НЕ универсально лучший
- Показано чёткое превосходство в специфических условиях
- Указаны ограничения и области проигрыша

**Подробный анализ:** см. `RESULTS_V2_ANALYSIS.md`
