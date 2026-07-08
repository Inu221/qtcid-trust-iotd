# Эксперименты по доверенному обнаружению и приоритизации системной проверки в IoT-сетях

Репозиторий содержит два воспроизводимых вычислительных сценария для оценки методов доверенного мониторинга распределённой IoT-сети:

- эксперимент по **приоритизации системной проверки узлов** на основе истории локальных решений;
- эксперимент по **сравнению методов Q-TCID и TA-QTCID**.

Оба сценария запускаются независимо и сохраняют результаты в каталог `results/`.

## Содержимое репозитория

### 1. Приоритизация системной проверки узлов
Этот сценарий предназначен для исследования ранжирования узлов при ограниченном бюджете аудита. Для него используется финальная реализация `game_ext/qtcid_repro/audit_prioritization_core_v2.py`.

Основные файлы:
- `game_ext/qtcid_repro/audit_prioritization_core_v2.py`
- `game_ext/qtcid_repro/experiments/audit_prioritization_study_article.py`
- `game_ext/qtcid_repro/experiments/visualization_article_ru.py`

Результаты сохраняются в каталог `results/audit_prioritization_article/`.

### 2. Сравнение Q-TCID и TA-QTCID
Этот сценарий предназначен для сравнения базового и trust-aware вариантов метода обнаружения компрометированных узлов.

Текущая реализация TA-QTCID содержит явный trust-aware механизм:
- пер-узловые доверительные веса `trust_weight[node_id]`;
- взвешенное голосование по формуле `sum(w_i * vote_i) / sum(w_i)`;
- порог принятия решения `theta`;
- обновление весов голосующих узлов через `eta_plus` и `eta_minus`;
- отдельный учёт обновлений по аудиту и по коллективному решению.

CMVI считается одинаково для Q-TCID и TA-QTCID:

```text
CMVI = Nfg + Nbr + Nam
```

где `Nfg` — ложные исключения корректных узлов, `Nbr` — сохранённые вредоносные узлы, `Nam` — события расхождения с аудитом. Дополнительные понижающие коэффициенты CMVI для TA-QTCID не используются.

Основные файлы:
- `game_ext/qtcid_repro/qtcid_core.py`
- `game_ext/qtcid_repro/ta_qtcid_core.py`
- `game_ext/qtcid_repro/experiments/final_qtcid_taqtcid_study.py`

Результаты сохраняются в каталог `results/final_qtcid_taqtcid_study/`.

## Требования

- Python 3.10+
- Linux / WSL / macOS

Установка окружения:

```bash
cd qtcid-trust-iotd
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Как воспроизвести эксперименты

### Эксперимент 1. Приоритизация системной проверки узлов

Полный расчёт с построением финальных русифицированных рисунков и таблицы:

```bash
cd qtcid-trust-iotd
source .venv/bin/activate
AUDIT_PRIORITIZATION_MODE=ARTICLE python -m game_ext.qtcid_repro.experiments.audit_prioritization_study_article
```

Быстрая пересборка только финальных рисунков и таблицы по уже рассчитанным данным:

```bash
cd qtcid-trust-iotd
source .venv/bin/activate
AUDIT_PRIORITIZATION_RENDER_ONLY=1 python -m game_ext.qtcid_repro.experiments.audit_prioritization_study_article
```

После выполнения будут созданы:
- детальные результаты в `results/audit_prioritization_article/detailed_results_article.csv` и `results/audit_prioritization_article/detailed_results_article.json`;
- финальные рисунки в `results/audit_prioritization_article/figures_article_final_ru/`;
- финальная таблица в `results/audit_prioritization_article/tables_article_final_ru/`.

### Эксперимент 2. Сравнение Q-TCID и TA-QTCID

Запуск полного сравнения:

```bash
cd qtcid-trust-iotd
source .venv/bin/activate
python -m game_ext.qtcid_repro.experiments.final_qtcid_taqtcid_study
```

После выполнения будут созданы таблицы и рисунки в каталоге `results/final_qtcid_taqtcid_study/`.

Основные таблицы:
- `table_qtcid_vs_taqtcid_detailed.csv` — агрегированные значения по всей сетке `Pa × TIDS`, включая `mean`, `std`, 95% CI для CMVI, paired t-test и Wilcoxon;
- `table_qtcid_vs_taqtcid_representative.csv` — компактная таблица для выбранных значений `TIDS`;
- `table_qtcid_vs_taqtcid_long_for_article.csv` — long-format таблица с колонкой `method` для вставки в статью;
- `table_absolute_cmvi_grid.csv` — абсолютные значения CMVI по сетке `Pa × TIDS`;
- `table_cmvi_decomposition.csv` — декомпозиция CMVI по компонентам `Nfg`, `Nbr`, `Nam`;
- `raw_paired_results.csv` — сырые paired Monte Carlo результаты по каждому прогону;
- `table_bvs_baseline.csv` — вспомогательная таблица базового BVS.

Основные рисунки для статьи:
- `figure_3_mttf_by_tids_new.png` и `.svg` — MTTF от интервала диагностики;
- `figure_4_cmvi_by_tids_new.png` и `.svg` — CMVI от интервала диагностики;
- `figure_5_delta_cmvi_new.png` и `.svg` — относительное изменение CMVI;
- `figure_6_average_delta_cmvi_new.png` и `.svg` — средние изменения CMVI по `TIDS` и `Pa`.

## Финальные рисунки для приоритизации системной проверки

Для финального набора публикационных иллюстраций используется единая цветовая схема:
- предлагаемый метод — зелёный;
- только текущие наблюдения — оранжевый;
- случайный выбор — серый;
- с историей без компонента стабильности — синий.

Финальный набор включает три основных рисунка:
- heatmap улучшения точности отбора предлагаемого метода относительно метода только по текущим наблюдениям;
- составной рисунок для главного сценария с тремя подграфиками: точность отбора, полнота выявления и доля ложных проверок шумных корректных узлов;
- ablation-график по вкладу компонентов истории.

## Структура каталогов результатов

```text
results/
├── audit_prioritization_article/
│   ├── detailed_results_article.csv
│   ├── detailed_results_article.json
│   ├── figures_article_final_ru/
│   └── tables_article_final_ru/
└── final_qtcid_taqtcid_study/
    ├── raw_paired_results.csv
    ├── table_qtcid_vs_taqtcid_detailed.csv
    ├── table_qtcid_vs_taqtcid_long_for_article.csv
    ├── table_absolute_cmvi_grid.csv
    ├── table_cmvi_decomposition.csv
    ├── figure_3_mttf_by_tids_new.png
    ├── figure_3_mttf_by_tids_new.svg
    ├── figure_4_cmvi_by_tids_new.png
    ├── figure_4_cmvi_by_tids_new.svg
    ├── figure_5_delta_cmvi_new.png
    ├── figure_5_delta_cmvi_new.svg
    ├── figure_6_average_delta_cmvi_new.png
    └── figure_6_average_delta_cmvi_new.svg
```

## Примечание по воспроизводимости

Во всех финальных сценариях параметры экспериментов и начальные значения генераторов случайных чисел зафиксированы в коде. Для сравнения Q-TCID и TA-QTCID используется paired design: оба метода запускаются на одинаковых `seed + run_id`.

Параметры финального сравнения:

```text
RUNS = 80
Pa = [0.0, 0.25, 0.5, 0.75, 1.0]
TIDS = [50, 100, 200, 350, 600, 1000, 1500]
```

Параметры доверия TA-QTCID:

```text
w0 = 1.0
w_min = 0.05
w_max = 2.0
eta_plus = 0.05
eta_minus = 0.30
theta = 0.5
```

Если требуется только пересобрать рисунки и таблицы без повторного Monte Carlo, используйте режим `AUDIT_PRIORITIZATION_RENDER_ONLY=1` для сценария приоритизации системной проверки.
