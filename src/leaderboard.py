import yaml
from typing import List, Dict, Any, Tuple
import time
from pathlib import Path
import json
from datetime import datetime
from src.equality_checker import MathEqualityChecker
from src.sampler import OaiSampler
from src.mat_boy import RussianMathEval
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import hashlib

class Leaderboard:
    def __init__(self, config_path: str, output_dir: str = "results", max_workers: int = 4):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        
        # Создаем директории
        self.details_dir = self.output_dir / "details"
        self.details_dir.mkdir(exist_ok=True)
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Загружаем конфиг и кэш
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_links = self.config.get('model_links', {})
        self.equality_checker = MathEqualityChecker()
        self.results_file = self.output_dir / "leaderboard_results.json"
        self.results = self._load_results()
        
    def _get_cache_key(self, model_name: str, system_prompt: str | None) -> str:
        """Генерирует ключ кэша на основе модели и промпта"""
        # Собираем все параметры, влияющие на результат
        cache_data = {
            'model_name': model_name,
            'system_prompt': system_prompt,
            'num_examples': self.config.get('num_examples'),
            'temperature': self.config.get('temperature'),
            'max_tokens': self.config.get('max_tokens'),
        }
        # Создаем хэш из параметров
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Dict | None:
        """Получает результат из кэша если он есть"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

    def _save_to_cache(self, cache_key: str, result: Dict):
        """Сохраняет результат в кэш"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)

    def _load_results(self) -> Dict:
        """Загружает существующие результаты"""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_results(self):
        """Сохраняет результаты в JSON"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def _save_detailed_results(self, model_name: str, results: List[Dict], timestamp: str):
        """Сохраняет детальные результаты для каждого примера"""
        model_dir = self.details_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        details = {
            "timestamp": timestamp,
            "model_name": model_name,
            "examples": []
        }
        
        for idx, result in enumerate(results):
            example_details = {
                "index": idx,
                "task": result.convo[0]["content"],
                "model_response": result.convo[1]["content"],
                "correct_answer": result.correct_answer,
                "extracted_answer": result.extracted_answer,
                "score": result.score,
                "tokens": result.tokens
            }
            details["examples"].append(example_details)
            
        # Сохраняем детальные результаты
        with open(model_dir / f"details_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
            
        # Генерируем markdown с детальными результатами
        md = f"# Detailed Results for {model_name}\n\n"
        md += f"Timestamp: {timestamp}\n\n"
        
        for example in details["examples"]:
            md += f"## Example {example['index'] + 1}\n\n"
            md += f"### Task\n{example['task']}\n\n"
            md += f"### Model Response\n{example['model_response']}\n\n"
            md += f"### Correct Answer\n{example['correct_answer']}\n\n"
            md += f"### Extracted Answer\n{example['extracted_answer']}\n\n"
            md += f"### Score\n{example['score']}\n\n"
            md += f"### Tokens Used\n{example['tokens']}\n\n"
            md += "---\n\n"
            
        with open(model_dir / f"details_{timestamp}.md", 'w', encoding='utf-8') as f:
            f.write(md)

    def evaluate_model(self, model_name: str, system_prompt: str = None) -> Dict[str, Any]:
        """Оценивает одну модель с использованием кэша"""
        # Генерируем ключ кэша
        cache_key = self._get_cache_key(model_name, system_prompt)
        
        # Проверяем кэш
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            if self.config.get('debug'):
                print(f"\nUsing cached result for {model_name}")
            return cached_result

        if self.config.get('debug'):
            print(f"\nEvaluating {model_name} (not found in cache)")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Обновляем конфиг для текущей модели
        self.config['model_list'] = [model_name]
        if system_prompt is not None:
            self.config[model_name]['system_prompt'] = system_prompt
            
        # Создаем временный конфиг
        temp_config_path = self.output_dir / f"temp_config_{model_name}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(self.config, f)
            
        try:
            # Инициализируем сэмплер и эвалюатор
            sampler = OaiSampler(str(temp_config_path))
            evaluator = RussianMathEval(
                equality_checker=self.equality_checker,
                num_examples=self.config.get('num_examples', None),
                debug=self.config.get('debug', False)
            )
            
            start_time = time.time()
            results = evaluator(sampler)
            evaluation_time = time.time() - start_time
            
            # Сохраняем детальные результаты
            self._save_detailed_results(model_name, results.results, timestamp)
            
            # Собираем метрики
            total_tokens = sum(r.tokens for r in results.results if hasattr(r, 'tokens'))
            
            # Формируем результат
            model_result = {
                "model_name": model_name,
                "score": results.score,
                "total_tokens": total_tokens,
                "evaluation_time": evaluation_time,
                "system_prompt": system_prompt,
                "timestamp": timestamp,
                "cache_key": cache_key,
                "config": {
                    "temperature": self.config.get('temperature', 0.0),
                    "max_tokens": self.config.get('max_tokens', 2048),
                    "num_examples": self.config.get('num_examples', None)
                }
            }
            
            # Сохраняем в кэш и общие результаты
            self._save_to_cache(cache_key, model_result)
            self.results[f"{model_name}_{timestamp}"] = model_result
            self._save_results()
            
            return model_result
            
        finally:
            # Удаляем временный конфиг
            temp_config_path.unlink(missing_ok=True)

    def evaluate_all_models(self, system_prompts: Dict[str, str] = None) -> None:
        """Оценивает все модели из конфига параллельно с использованием кэша"""
        if system_prompts is None:
            system_prompts = {}
            
        # Создаем список аргументов для параллельной обработки
        eval_args = [
            (model_name, system_prompts.get(model_name))
            for model_name in self.config['model_list']
        ]
        
        # Фильтруем только те модели, которых нет в кэше
        uncached_args = []
        for args in eval_args:
            cache_key = self._get_cache_key(args[0], args[1])
            if self._get_cached_result(cache_key) is None:
                uncached_args.append(args)
        
        if uncached_args:
            print(f"\nEvaluating {len(uncached_args)} uncached models...")
            # Используем ThreadPoolExecutor только для некэшированных моделей
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                list(tqdm(
                    executor.map(self.evaluate_model_parallel, uncached_args),
                    total=len(uncached_args),
                    desc="Evaluating models"
                ))
        else:
            print("\nAll models found in cache!")
        
        # Загружаем результаты для всех моделей (включая кэшированные)
        for model_name, system_prompt in eval_args:
            self.evaluate_model(model_name, system_prompt)

    def generate_markdown(self) -> str:
        """Генерирует markdown с результатами"""
        md = "# Math Evaluation Leaderboard\n\n"
        md += f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Заголовок таблицы
        md += "| Model | Score | Tokens Used | System Prompt | Evaluation Time | Details |\n"
        md += "|-------|--------|-------------|---------------|----------------|----------|\n"
        
        # Сортируем результаты по score
        sorted_results = sorted(
            self.results.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Добавляем строки таблицы
        for result in sorted_results:
            system_prompt = result['system_prompt'] or 'None'
            if len(system_prompt) > 30:
                system_prompt = system_prompt[:27] + "..."
                
            details_link = f"[Details](details/{result['model_name']}/details_{result['timestamp']}.md)"
                
            md += f"| {result['model_name']} "
            md += f"| {result['score']:.3f} "
            md += f"| {result['total_tokens']} "
            md += f"| {system_prompt} "
            md += f"| {result['evaluation_time']:.1f}s "
            md += f"| {details_link} |\n"
            
        # Сохраняем markdown
        with open(self.output_dir / "leaderboard.md", 'w') as f:
            f.write(md)
            
        return md

def main():
    # Пример использования
    leaderboard = Leaderboard('configs/run.yaml')
    
    # Определяем разные system prompts для моделей
    system_prompts = {
        "gpt-4-1106-preview": "You are a helpful math assistant. Answer in Russian.",
        "gpt-3.5-turbo-0125": "Solve math problems step by step. Answer in Russian.",
        "gpt-4o-mini": "You are a math expert. Provide detailed solutions in Russian.",
    }
    
    # Оцениваем все модели с разными system prompts
    leaderboard.evaluate_all_models(system_prompts)
    
    # Генерируем markdown с результатами
    md = leaderboard.generate_markdown()
    print("Leaderboard generated!")
    print(md)

if __name__ == "__main__":
    main()
