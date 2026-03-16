# Q-TCID / TA-QTCID — Эксперименты для статьи

Репозиторий содержит реализацию и сравнительный эксперимент двух методов обнаружения компрометированных узлов в IoT-сетях:

- **Q-TCID** — базовый метод-аналог на основе Q-обучения с голосованием (реализация по Wang et al.)
- **TA-QTCID** (Trust-Aware Q-TCID) — авторский метод, расширяющий Q-TCID динамической моделью доверия узлов

## Ключевая идея

При сопоставимых значениях MTTF и точности обнаружения TA-QTCID стабильно снижает метрику CMVI (Cumulative Misclassification Vulnerability Index) на **3–12%** во всём диапазоне параметров атаки $P_a \in [0, 1]$ и интервала диагностики $T_{IDS} \in [50, 1500]$.

## Структура

```
game_ext/qtcid_repro/
├── qtcid_core.py          # Реализация Q-TCID
├── ta_qtcid_core.py       # Реализация TA-QTCID (авторский метод)
├── types.py               # Общие типы данных
├── utils.py               # Вспомогательные функции
├── mitchell/              # Базовые формулы (энергия, голосование) по Mitchell
│   ├── energy.py
│   └── voting.py
├── wang/                  # Базовая система BVS (аналог) по Wang
│   ├── bvs_core.py
│   └── game.py
└── experiments/
    └── final_qtcid_taqtcid_study.py  # Основной сравнительный эксперимент

results/final_qtcid_taqtcid_study/
├── fig_cmvi_qtcid_vs_taqtcid.png               # CMVI: Q-TCID vs TA-QTCID по Pa
├── fig_mttf_qtcid_vs_taqtcid.png               # MTTF: Q-TCID vs TA-QTCID по Pa
├── fig_cmvi_improvement_pct.png                 # Улучшение CMVI, %
├── fig_avg_cmvi_improvement_representative.png  # Среднее улучшение (TIDS=200,600,1000)
├── table_qtcid_vs_taqtcid_detailed.csv         # Полные результаты (35 комбинаций Pa×TIDS)
├── table_qtcid_vs_taqtcid_representative.csv    # Репрезентативная выборка
├── table_bvs_baseline.csv                       # Базовые результаты BVS
└── table_all_metrics.html                       # Сводная таблица для вставки в Word
```

## Параметры эксперимента

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

## Запуск

```bash
python3 -m game_ext.qtcid_repro.experiments.final_qtcid_taqtcid_study
```

Результаты сохраняются в `results/final_qtcid_taqtcid_study/`.

## Зависимости

```bash
pip install -r requirements.txt
```
