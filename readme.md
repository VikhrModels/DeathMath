# Deathmath benchmark, most codebase from openai simpleeval

Это бенчмарк для оценки качества языковых моделей на математических задачах.

=== LEADERBOARD ===

| Model | Score | Tokens Used | System Prompt | Evaluation Time | Dataset | Details 
|-------|--------|-------------|---------------|----------------|---------|----------
| o3-mini-2025-01-31 | 0.400 | 0 | You are a helpful math assi... | 14.8s | RussianMath | [Details](details/o3-mini-2025-01-31/details_20250408_072911.md) 
| gpt-4o | 0.332 | 0 | You are a helpful math assi... | 486.6s | RussianMath | [Details](details/gpt-4o/details_20250409_235721.md) 
| gpt-4o-mini | 0.300 | 0 | You are a helpful math assi... | 504.3s | RussianMath | [Details](details/gpt-4o-mini/details_20250409_235721.md) 
| GigaChat-2-Max | 0.205 | 83643 | You are a helpful math assi... | 418.1s | RussianMath | [Details](details/GigaChat-2-Max/details_20250410_154315.md) 
| GigaChat-2-Pro | 0.195 | 87907 | You are a helpful math assi... | 374.4s | RussianMath | [Details](details/GigaChat-2-Pro/details_20250410_154315.md) 
| GigaChat-Max | 0.158 | 91274 | You are a helpful math assi... | 512.1s | RussianMath | [Details](details/GigaChat-Max/details_20250410_154315.md) 
| GigaChat-2 | 0.089 | 73978 | You are a helpful math assi... | 221.2s | RussianMath | [Details](details/GigaChat-2/details_20250410_154315.md)

## Поддерживаемые датасеты

1. **RussianMath** - задачи по математике на русском языке (основной датасет)
2. **MathDemon_Demidovich** - подмножества задач из учебника Демидовича, включая:
   - Approximation_by_Polynomials
   - Continuous_Functions
   - Convex_Functions
   - Differentiation
   - Improper_Integrals
   - Infinite_Series
   - Integration
   - Sequences_and_Limits
   - Series_of_Functions

## Запуск

### Базовый запуск (все датасеты)

```bash
python runner.py
```

### Выбор конкретного датасета

```bash
python runner.py --dataset russianmath  # Только датасет RussianMath
python runner.py --dataset mathdemon    # Только датасет MathDemon_Demidovich
```

### Другие параметры

```bash
python runner.py --no-cache       # Игнорировать кэш и повторно выполнить оценку
python runner.py --max-workers 8  # Установить количество параллельных обработчиков
python runner.py --config path/to/config.yaml  # Указать альтернативный конфиг
```

### Справка по параметрам

```bash
python runner.py --help
```

## Конфигурация

Настройка выполняется через файлы YAML в директории `configs/`:

```yaml
configs/run.yaml
```

## Генерация таблицы лидеров

После запуска оценки автоматически будет сгенерирована таблица лидеров. 
Она сохраняется в `results/leaderboard.md`.