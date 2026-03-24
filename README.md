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
```

## Примечание по воспроизводимости

Во всех финальных сценариях параметры экспериментов и начальные значения генераторов случайных чисел зафиксированы в коде. Если требуется только пересобрать рисунки и таблицы без повторного Monte Carlo, используйте режим `AUDIT_PRIORITIZATION_RENDER_ONLY=1` для сценария приоритизации системной проверки.
