import yaml
from src.equality_checker import MathEqualityChecker
from src.leaderboard import Leaderboard
import argparse
from pathlib import Path
import sys
import os

def main():
    # Установка кодировки вывода в UTF-8
    if sys.platform == 'win32':
        os.system('chcp 65001')  # Установка кодировки UTF-8 для Windows консоли
    
    parser = argparse.ArgumentParser(
        description='Оценка языковых моделей на математических задачах',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python runner.py                        # Запустить оценку на всех датасетах (по умолчанию)
  python runner.py --dataset russianmath  # Запустить только на датасете RussianMath
  python runner.py --dataset mathdemon    # Запустить только на датасете MathDemon_Demidovich
  python runner.py --no-cache             # Игнорировать кэш и переоценить все модели
  python runner.py --max-workers 8        # Использовать 8 параллельных потоков
        """
    )
    parser.add_argument('--max-workers', type=int, default=4,
                      help='Максимальное количество параллельных потоков (по умолчанию: 4)')
    parser.add_argument('--config', type=str, default='configs/run.yaml',
                      help='Путь к файлу конфигурации (по умолчанию: configs/run.yaml)')
    parser.add_argument('--no-cache', action='store_true',
                      help='Игнорировать кэш и переоценить все модели')
    parser.add_argument('--dataset', choices=['all', 'russianmath', 'mathdemon'], default='all',
                      help='Выбор датасета для оценки: all (все), russianmath, mathdemon (по умолчанию: all)')
    args = parser.parse_args()

    # Загружаем конфиг
    with open(args.config, 'r', encoding='utf-8') as f:
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
    leaderboard = Leaderboard(args.config, max_workers=args.max_workers)
    
    # Определяем system prompts для моделей
    system_prompts = {
        model: config.get(model, {}).get('system_prompt')
        for model in config['model_list']
    }
    
    # Запуск оценки в зависимости от выбранного датасета
    if args.dataset == 'all':
        print("\nЗапуск оценки на всех датасетах (RussianMath и MathDemon_Demidovich)")
        leaderboard.evaluate_all_models(system_prompts)
        leaderboard.evaluate_math_demon_subsets()
    elif args.dataset == 'russianmath':
        print("\nЗапуск оценки только на датасете RussianMath")
        leaderboard.evaluate_all_models(system_prompts)
    elif args.dataset == 'mathdemon':
        print("\nЗапуск оценки только на датасете MathDemon_Dемидович")
        leaderboard.evaluate_math_demon_subsets()
    
    # Вычисляем комбинированные оценки для моделей
    if args.dataset == 'all':
        print("\nВычисление комбинированной оценки по всем датасетам")
        leaderboard.calculate_combined_scores()
    
    # Генерируем и выводим лидерборд
    TERMINAL_WIDTH = 100  # Примерная ширина терминала
    header = " LEADERBOARD "
    padding = "=" * ((TERMINAL_WIDTH - len(header)) // 2)
    print(f"\n{padding}{header}{padding}\n")
    
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

    print(f"\nДетальные результаты сохранены в: {leaderboard.output_dir}")

if __name__ == "__main__":
    main()