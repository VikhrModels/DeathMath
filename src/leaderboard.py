import yaml
from typing import List, Dict, Any
import time
from pathlib import Path
import json
from datetime import datetime
from src.equality_checker import MathEqualityChecker
from src.sampler import OaiSampler
from src.mat_boy import RussianMathEval

class Leaderboard:
    def __init__(self, config_path: str, output_dir: str = "results"):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем директории для детальных результатов
        self.details_dir = self.output_dir / "details"
        self.details_dir.mkdir(exist_ok=True)
        
        # Загружаем конфиг
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Получаем ссылки на документацию из конфига
        self.model_links = self.config.get('model_links', {})
            
        # Создаем equality checker
        self.equality_checker = MathEqualityChecker()
        
        # Загружаем результаты если есть
        self.results_file = self.output_dir / "leaderboard_results.json"
        self.results = self._load_results()

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
        """Оценивает одну модель"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Получаем количество примеров из конфига
        num_examples = self.config.get('num_examples', 5)  # По умолчанию 5 примеров
        
        # Обновляем конфиг для текущей модели
        self.config['model_list'] = [model_name]
        if system_prompt is not None:
            self.config[model_name]['system_prompt'] = system_prompt
            
        # Создаем временный конфиг
        temp_config_path = self.output_dir / f"temp_config_{model_name}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(self.config, f)
            
        # Инициализируем сэмплер и эвалюатор
        sampler = OaiSampler(str(temp_config_path))
        evaluator = RussianMathEval(
            equality_checker=self.equality_checker,
            num_examples=num_examples,  # Передаем количество примеров
            debug=self.config.get('debug', False)
        )
        
        # Замеряем время и токены
        start_time = time.time()
        total_tokens = 0
        
        # Запускаем оценку
        results = evaluator(sampler)
        
        # Сохраняем детальные результаты
        self._save_detailed_results(model_name, results.results, timestamp)
        
        # Собираем метрики
        evaluation_time = time.time() - start_time
        
        # Получаем использованные токены
        for result in results.results:
            if hasattr(result, 'tokens'):
                total_tokens += result.tokens
        
        # Формируем результат
        model_result = {
            "model_name": model_name,
            "score": results.score,
            "total_tokens": total_tokens,
            "evaluation_time": evaluation_time,
            "system_prompt": system_prompt,
            "timestamp": timestamp,
            "config": {
                "temperature": self.config.get('temperature', 0.0),
                "max_tokens": self.config.get('max_tokens', 2048),
                "num_examples": self.config.get('num_examples', None)
            }
        }
        
        # Сохраняем результат
        self.results[f"{model_name}_{timestamp}"] = model_result
        self._save_results()
        
        # Удаляем временный конфиг
        temp_config_path.unlink()
        
        return model_result

    def evaluate_all_models(self, system_prompts: Dict[str, str] = None) -> None:
        """Оценивает все модели из конфига"""
        if system_prompts is None:
            system_prompts = {}
            
        for model_name in self.config['model_list']:
            system_prompt = system_prompts.get(model_name)
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
