# Критический аудит метода приоритизации системной проверки

## ПРОБЛЕМЫ ТЕКУЩЕЙ РЕАЛИЗАЦИИ

### ❌ ПРОБЛЕМА 1: ПРЯМАЯ УТЕЧКА GROUND TRUTH В DISAGREEMENT

**Файл:** `audit_prioritization_core.py:267`

```python
def compute_disagreement_index(self, nid: int) -> float:
    hist = self.node_histories[nid]
    if not hist.disagreement_events:
        # Базовая оценка на основе current suspicion
        return 0.3 if hist.is_bad else 0.1  # ← УТЕЧКА!
```

**Последствие:** Current-only метод получает прямой доступ к истинному состоянию узла через `hist.is_bad`. Это делает его почти идеальным оракулом.

---

### ❌ ПРОБЛЕМА 2: DISAGREEMENT ФОРМИРУЕТСЯ ИЗ GROUND TRUTH

**Файл:** `audit_prioritization_core.py:373-381`

```python
for nid in active:
    hist = self.node_histories[nid]
    if hist.is_bad:  # ← УТЕЧКА!
        # Bad узел: высокое disagreement
        disagreement_val = 0.5 + 0.3 * self.rng.random()
    else:
        # Good узел: низкое disagreement
        if self.rng.random() < p_fp_ids:
            disagreement_val = 0.3 + 0.2 * self.rng.random()
        else:
            disagreement_val = 0.05 + 0.1 * self.rng.random()
```

**Последствие:**
- Bad узлы ВСЕГДА получают disagreement [0.5, 0.8]
- Good узлы ВСЕГДА получают disagreement [0.05, 0.15] (с малой вероятностью [0.3, 0.5])
- Нет реальной неопределённости, нет шумных benign узлов, нет intermittent malicious

---

### ❌ ПРОБЛЕМА 3: PRECISION ≡ HIT_RATE

**Файл:** `audit_prioritization_core.py:534-536`

```python
hit_rate = safe_div(audits_on_bad, total_audits)
wasted_rate = safe_div(audits_on_good, total_audits)
precision = hit_rate  # ← НЕПРАВИЛЬНО!
```

**Последствие:** Precision и hit_rate - это одна и та же метрика!

**Правильные определения:**
- **Precision** = TP / (TP + FP) = доля правильно выбранных среди выбранных для аудита
- **Hit Rate (Recall)** = TP / (TP + FN) = доля обнаруженных bad узлов от всех bad

Или:
- **Audit Precision** = (audits_on_bad) / (total_audits) - доля bad среди выбранных
- **Hit Rate at Budget** = (audits_on_bad) / (total_bad_nodes) - доля bad, попавших в аудит

---

### ❌ ПРОБЛЕМА 4: ОТСУТСТВИЕ ШУМНЫХ И INTERMITTENT УЗЛОВ

Все узлы делятся строго на:
- Good: низкое disagreement
- Bad: высокое disagreement

**Отсутствуют:**
- `benign_noisy`: нормальный узел с кратковременными аномалиями
- `malicious_intermittent`: вредоносный узел с эпизодическим поведением
- `malicious_persistent`: вредоносный узел с устойчивым поведением

**Последствие:** Нет сценариев, где история действительно нужна.

---

### ❌ ПРОБЛЕМА 5: ОТСУТСТВИЕ РЕАЛЬНОГО ЛОКАЛЬНОГО ГОЛОСОВАНИЯ

Disagreement симулируется напрямую из ground truth, а не вычисляется из:
- Реальных голосов соседей
- Ошибок локальных оценок
- Вредоносного голосования

**Последствие:** Модель нереалистична для распределённой системы.

---

## ПОЧЕМУ CURRENT-ONLY ПОЧТИ ИДЕАЛЕН

1. **Прямой доступ к ground truth** через `hist.is_bad`
2. **Детерминированное disagreement**: bad ↔ высокое, good ↔ низкое
3. **Нет шума и неопределённости** в локальных наблюдениях
4. **Нет intermittent attacks**, которые трудно поймать без истории

**Результат:**
- Current-only получает hit_rate=1.0 при budget=3
- History компоненты не дают преимущества
- Метод не доказывает полезность истории

---

## ПОЧЕМУ PRECISION ≈ HIT_RATE

```python
precision = audits_on_bad / total_audits
hit_rate = audits_on_bad / total_audits
```

Обе метрики вычисляются одинаково! Правильно было бы:
```python
precision = audits_on_bad / total_audits  # доля bad среди выбранных
recall = audits_on_bad / total_bad_nodes  # доля bad, попавших в аудит
```

---

## ПЛАН ИСПРАВЛЕНИЙ

### ✅ ШАГ 1: Убрать утечку ground truth
- Disagreement формируется из реальных локальных голосов
- Нет прямого доступа к `is_bad` при вычислении scores

### ✅ ШАГ 2: Реализовать реальное локальное голосование
- У каждого узла есть соседи-наблюдатели
- Каждый наблюдатель даёт локальную оценку с ошибкой
- Disagreement = функция разброса голосов
- Anomaly count = число тревожных сигналов

### ✅ ШАГ 3: Добавить типы узлов
- `benign_stable`: нормальный, низкий шум
- `benign_noisy`: нормальный, но шумный
- `malicious_intermittent`: вредоносный, эпизодический
- `malicious_persistent`: вредоносный, устойчивый

### ✅ ШАГ 4: Усилить history-based method
- EMA mismatch memory
- Stability index для длительно корректных узлов
- Persistence penalty для повторяющихся конфликтов
- Decay factor для старой истории

### ✅ ШАГ 5: Исправить метрики
- Развести precision и hit_rate
- Добавить: delay_to_first_audit, false_attention_rate, intermittent_detection_rate

### ✅ ШАГ 6: Новые сценарии
- Low noise / persistent attacks
- Medium noise / intermittent attacks
- High noise / intermittent attacks
- Mixed population

### ✅ ШАГ 7: Анализ чувствительности
- Sweep по budget, noise level, intermittent probability
- Heatmaps и ablation studies

---

## ОЖИДАЕМЫЙ РЕЗУЛЬТАТ

После исправлений:
- **Current-only** останется сильным baseline, но не оракулом
- **History-based** будет превосходить в шумных и intermittent сценариях
- **Random** будет проигрывать везде
- Появится clear и reproducible класс сценариев, где история полезна

---

## ЧЕСТНЫЙ ВЫВОД

Предлагаемый метод не обязан побеждать везде, но должны появиться:
1. Сценарии с **чётким превосходством** history-based
2. Сценарии с **comparable** результатами (простые случаи)
3. Понимание, **когда и почему** история помогает

Эти сценарии должны быть **естественными** для распределённых беспилотных систем с шумными локальными оценками.
