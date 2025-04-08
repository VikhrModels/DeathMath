# Deathmath benchmark, most codebase from openai simpleeval


=== LEADERBOARD ===

| Model | Score | Tokens Used | System Prompt | Evaluation Time | Details 
|-------|--------|-------------|---------------|----------------|----------
| o3-mini-2025-01-31 | 0.400 | 0 | You are a helpful math assi... | 770.4s | [Details](details/o3-mini-2025-01-31/details_20250408_083654.md) 
| gpt-4o | 0.395 | 0 | You are a helpful math assi... | 3282.2s | [Details](details/gpt-4o/details_20250408_074210.md) 
| gpt-4o-mini | 0.274 | 0 | You are a helpful math assi... | 609.1s | [Details](details/gpt-4o-mini/details_20250408_073159.md) 


## Run

```bash
python runner.py
```

## Config

```yaml
configs/run.yaml
```

## Leaderboard

```bash
python leaderboard.py
```