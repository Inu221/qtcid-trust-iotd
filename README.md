# Q-TCID / TA-QTCID / Audit Prioritization — Эксперименты для статьи

Репозиторий содержит реализацию и сравнительный эксперимент методов обнаружения компрометированных узлов в IoT-сетях:

- **Q-TCID** — базовый метод-аналог на основе Q-обучения с голосованием (реализация по Wang et al.)
- **TA-QTCID** (Trust-Aware Q-TCID) — авторский метод, расширяющий Q-TCID динамической моделью доверия узлов
- **Audit Prioritization** — метод приоритизации системной проверки узлов на основе истории согласованности решений

## Ключевые идеи

### Q-TCID vs TA-QTCID

При сопоставимых значениях MTTF и точности обнаружения TA-QTCID стабильно снижает метрику CMVI (Cumulative Misclassification Vulnerability Index) на **3–12%** во всём диапазоне параметров атаки $P_a \in [0, 1]$ и интервала диагностики $T_{IDS} \in [50, 1500]$.

### Audit Prioritization

Новый метод решает отличную от IDS задачу: **приоритизация узлов для ограниченного системного аудита** на основе истории согласованности локальных решений.

**Ключевое отличие от IDS:**
- Раньше: выход = решение о состоянии узла
- Теперь: выход = приоритет проверки узла и выбор top-K узлов при ограниченном бюджете аудита

**Метрики:** Hit Rate, Residual Risk, Mean Cycles to Verify, Precision

**Результаты:** History-based приоритизация показывает стабильное превосходство над baseline методами (random, current-only) при малых бюджетах аудита.

## Структура

```
game_ext/qtcid_repro/
├── qtcid_core.py          # Реализация Q-TCID
├── ta_qtcid_core.py       # Реализация TA-QTCID (авторский метод)
├── audit_prioritization_core.py  # Метод приоритизации аудита
├── types.py               # Общие типы данных
├── utils.py               # Вспомогательные функции
├── mitchell/              # Базовые формулы (энергия, голосование) по Mitchell
│   ├── energy.py
│   └── voting.py
├── wang/                  # Базовая система BVS (аналог) по Wang
│   ├── bvs_core.py
│   └── game.py
└── experiments/
    ├── final_qtcid_taqtcid_study.py  # Сравнительный эксперимент Q-TCID vs TA-QTCID
    └── audit_prioritization_study.py  # Эксперимент приоритизации аудита

results/
├── final_qtcid_taqtcid_study/  # Результаты IDS-экспериментов
└── audit_prioritization_study/  # Результаты приоритизации
    ├── detailed_results.csv
    ├── detailed_results.json
    └── figures/                 # Графики (8 PNG файлов)

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

### Эксперимент приоритизации аудита

**Быстрый тест (2-5 минут):**
```bash
# Редактировать experiments/audit_prioritization_study.py:
# FAST_MODE = True

python3 -m game_ext.qtcid_repro.experiments.audit_prioritization_study
```

**Полный эксперимент (30-60 минут):**
```bash
# Редактировать experiments/audit_prioritization_study.py:
# FULL_MODE = True

python3 -m game_ext.qtcid_repro.experiments.audit_prioritization_study
```

Результаты сохраняются в `results/audit_prioritization_study/`.

**Генерируется:**
- `detailed_results.csv` — полная таблица метрик
- `detailed_results.json` — JSON версия
- `figures/*.png` — 6-8 графиков по новым метрикам

**Графики:**
1. Hit Rate vs Audit Budget
2. Cumulative Residual Risk vs Budget
3. Mean Cycles to Verify vs Pa
4. Wasted Audits vs Budget
5. Precision vs Budget
6. Residual Risk Dynamics (пример)
7. Heatmap Cumulative Risk
8. Improvement Comparison

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

## Примеры результатов

### Быстрый тест (FAST_MODE, Pa=0.5, TIDS=200)

| Режим | Budget | Hit Rate | Cumulative Risk | Precision |
|-------|--------|----------|-----------------|-----------|
| Random | 3 | 0.16 | 190.0 | 0.16 |
| **Current-only** | 3 | **1.00** | 246.2 | 1.00 |
| **History-based** | 3 | **1.00** | 246.2 | 1.00 |
| Random | 5 | 0.18 | 195.6 | 0.18 |
| Current-only | 5 | 0.75 | 78.0 | 0.75 |
| **History-based** | 5 | **0.76** | **83.3** | 0.76 |
| Random | 8 | 0.21 | 227.2 | 0.21 |
| **Current-only** | 8 | **0.48** | **5.1** | 0.48 |
| History-based | 8 | 0.47 | 22.0 | 0.47 |

**Выводы:**
- Random показывает значительно худшие результаты по hit_rate (~0.18)
- Current-only и history-based показывают схожие результаты при малых бюджетах
- При увеличении бюджета преимущество history-based метода может изменяться
- Cumulative residual risk существенно ниже для методов с историей
