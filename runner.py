import yaml
from src.equality_checker import MathEqualityChecker
from src.leaderboard import Leaderboard
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run model evaluation')
    parser.add_argument('--max-workers', type=int, default=4,
                      help='Maximum number of parallel workers (default: 4)')
    parser.add_argument('--config', type=str, default='configs/run.yaml',
                      help='Path to config file (default: configs/run.yaml)')
    parser.add_argument('--no-cache', action='store_true',
                      help='Ignore cache and re-evaluate all models')
    args = parser.parse_args()

    # Загружаем конфиг
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Если указан --no-cache, отключаем использование кэша
    if args.no_cache:
        # Очищаем директорию с кэшем
        cache_dir = Path("results/cache")
        if cache_dir.exists():
            for cache_file in cache_dir.glob("*.json"):
                cache_file.unlink()

    # Создаем equality checker
    equality_checker = MathEqualityChecker()

    # Создаем и обновляем лидерборд
    leaderboard = Leaderboard('configs/run.yaml')
    
    # Определяем system prompts для моделей
    system_prompts = {
        model: config.get(model, {}).get('system_prompt')
        for model in config['model_list']
    }
    
    # Оцениваем все модели
    leaderboard.evaluate_all_models(system_prompts)
    
    # Генерируем и выводим лидерборд
    print("\n=== LEADERBOARD ===\n")
    md = leaderboard.generate_markdown()
    
    # Выводим упрощенную версию лидерборда в консоль
    lines = md.split('\n')
    for line in lines:
        if line.startswith('|'):
            # Убираем ссылки [Details](path) из вывода в консоль
            cleaned_line = line.split('|')
            if len(cleaned_line) > 5:  # Проверяем, что это строка с данными
                cleaned_line = cleaned_line[:-1]  # Убираем последнюю колонку с ссылками
            print('|'.join(cleaned_line))

    print(f"\nDetailed results saved in: {leaderboard.output_dir}")

if __name__ == "__main__":
    main() 