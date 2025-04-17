import yaml
from typing import List, Dict, Any, Set
import time
from pathlib import Path
import json
from datetime import datetime
from src.equality_checker import MathEqualityChecker
from src.sampler import OaiSampler
from src.mat_boy import RussianMathEval, MathDemonEval
from src.types import SingleEvalResult
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tqdm.auto import tqdm
import hashlib
import signal
import sys


class Leaderboard:
    def __init__(
        self, config_path: str, output_dir: str = "results", max_workers: int = 4
    ):
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
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.model_links = self.config.get("model_links", {})
        self.equality_checker = MathEqualityChecker()
        self.results_file = self.output_dir / "leaderboard_results.json"
        self.results = self._load_results()

    def _get_cache_key(self, model_name: str, system_prompt: str | None) -> str:
        """Генерирует ключ кэша на основе модели и промпта"""
        # Используем безопасное имя модели для кэша
        safe_model_name = model_name.replace("/", "_")
        cache_data = {
            "model_name": safe_model_name,
            "system_prompt": system_prompt,
            "num_examples": self.config.get("num_examples"),
            "temperature": self.config.get("temperature"),
            "max_tokens": self.config.get("max_tokens"),
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Dict | None:
        """Получает результат из кэша если он есть"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                return json.load(f)
        return None

    def _save_to_cache(self, cache_key: str, result: Dict):
        """Сохраняет результат в кэш"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2)

    def _load_results(self) -> Dict:
        """Загружает существующие результаты и кэш"""
        results = {}

        # Загружаем все результаты из кэша
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                with open(cache_file, "r") as f:
                    cached_result = json.load(f)
                    model_name = cached_result["model_name"]
                    timestamp = cached_result["timestamp"]
                    results[f"{model_name}_{timestamp}"] = cached_result

        # Если есть файл с результатами, добавляем их тоже
        if self.results_file.exists():
            with open(self.results_file, "r") as f:
                file_results = json.load(f)
                results.update(file_results)

        return results

    def _save_results(self):
        """Сохраняет все результаты"""
        # Сохраняем в основной файл результатов
        with open(self.results_file, "w") as f:
            json.dump(self.results, f, indent=2)

    def _save_detailed_results(
        self,
        model_name: str,
        results: List[SingleEvalResult],
        timestamp: str,
        dataset: str = None,
    ):
        """Сохраняет детальные результаты для модели с указанием датасета"""
        # Создаем безопасное имя директории
        safe_model_name = model_name.replace("/", "_")
        model_dir = self.details_dir / safe_model_name
        model_dir.mkdir(exist_ok=True)

        # Добавляем суффикс датасета к имени файла если он указан
        file_suffix = f"_{dataset}" if dataset else ""

        # Сохраняем результаты в JSON
        details_file = model_dir / f"details_{timestamp}{file_suffix}.json"
        with open(details_file, "w", encoding="utf-8") as f:
            json.dump(
                results, f, indent=2, default=lambda x: x.__dict__, ensure_ascii=False
            )

        # Создаем и сохраняем markdown-отчет
        markdown_file = model_dir / f"details_{timestamp}{file_suffix}.md"
        markdown_content = self._generate_markdown_report(
            model_name, results, timestamp, dataset
        )
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        # Если у нас есть и по математике, и по физике отчеты для этой модели, объединяем их
        if dataset == "RussianPhysics":
            math_report = model_dir / f"details_{timestamp}.md"
            if math_report.exists():
                self._combine_detailed_reports(model_name, timestamp)

        return markdown_file

    def _generate_markdown_report(
        self,
        model_name: str,
        results: List[SingleEvalResult],
        timestamp: str,
        dataset: str = None,
    ) -> str:
        """Генерирует markdown-отчет с детальными результатами для модели"""
        # Добавляем информацию о датасете если она указана
        dataset_info = f" - {dataset}" if dataset else ""

        md = f"# Detailed Results for {model_name}{dataset_info}\n\n"
        md += f"Timestamp: {timestamp}\n\n"

        # Считаем общий скор
        correct_count = sum(1 for r in results if hasattr(r, "score") and r.score == 1)
        total_count = len(results)
        overall_score = correct_count / total_count if total_count > 0 else 0

        md += "## Summary\n\n"
        md += f"- **Dataset**: {dataset or 'RussianMath'}\n"
        md += f"- **Total examples**: {total_count}\n"
        md += f"- **Correct answers**: {correct_count}\n"
        md += f"- **Score**: {overall_score:.3f}\n\n"

        md += "---\n\n"

        for i, result in enumerate(results, 1):
            md += f"## Example {i}\n\n"

            # Добавляем задачу и ответ модели из convo, если он есть
            if hasattr(result, "convo") and result.convo:
                for message in result.convo:
                    if message.get("role") == "user":
                        md += f"### Task\n{message.get('content', '')}\n\n"
                    elif message.get("role") == "assistant":
                        md += f"### Model Response\n{message.get('content', '')}\n\n"

            # Добавляем правильный ответ
            if hasattr(result, "correct_answer") and result.correct_answer:
                md += f"### Correct Answer\n{result.correct_answer}\n\n"

            # Добавляем извлеченный ответ
            if (
                hasattr(result, "extracted_answer")
                and result.extracted_answer is not None
            ):
                md += f"### Extracted Answer\n{result.extracted_answer}\n\n"

            # Добавляем оценку
            if hasattr(result, "score") and result.score is not None:
                md += f"### Score\n{result.score}\n\n"

            # Добавляем количество токенов
            if hasattr(result, "tokens"):
                md += f"### Tokens Used\n{result.tokens}\n\n"

            md += "---\n\n"

        return md

    def evaluate_model(
        self, model_name: str, system_prompt: str = None
    ) -> Dict[str, Any]:
        """Оценивает одну модель"""
        cache_key = self._get_cache_key(model_name, system_prompt)
        cached_result = self._get_cached_result(cache_key)

        if cached_result is not None:
            if self.config.get("debug"):
                print(f"\nUsing cached result for {model_name}")
            return cached_result

        if self.config.get("debug"):
            print(f"\nEvaluating {model_name}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace("/", "_")

        # Создаем временный конфиг
        temp_config = self.config.copy()
        temp_config["model_list"] = [model_name]
        if system_prompt is not None:
            temp_config[model_name]["system_prompt"] = system_prompt

        temp_config_path = self.output_dir / f"temp_config_{safe_model_name}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(temp_config, f)

        try:
            sampler = OaiSampler(str(temp_config_path))
            evaluator = RussianMathEval(
                equality_checker=self.equality_checker,
                num_examples=self.config.get("num_examples", None),
                debug=self.config.get("debug", False),
            )

            start_time = time.time()
            results = evaluator(sampler)
            evaluation_time = time.time() - start_time

            # Сохраняем детальные результаты
            self._save_detailed_results(model_name, results.results, timestamp)

            total_tokens = sum(
                r.tokens for r in results.results if hasattr(r, "tokens")
            )

            model_result = {
                "model_name": model_name,  # Сохраняем оригинальное имя
                "score": results.score,
                "total_tokens": total_tokens,
                "evaluation_time": evaluation_time,
                "system_prompt": system_prompt,
                "timestamp": timestamp,
                "cache_key": cache_key,
            }

            # Сохраняем в кэш
            self._save_to_cache(cache_key, model_result)

            # Используем оригинальное имя модели для ключа результатов
            self.results[f"{model_name}_{timestamp}"] = model_result
            self._save_results()

            return model_result

        finally:
            temp_config_path.unlink(missing_ok=True)

    def evaluate_model_parallel(self, args: tuple) -> Dict[str, Any]:
        """Оценивает одну модель (для использования в ThreadPoolExecutor)"""
        model_name, system_prompt = args
        return self.evaluate_model(model_name, system_prompt)

    def _get_measured_models(self) -> Set[str]:
        """Получает список уже измеренных моделей из кэша"""
        measured_models = set()
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                    measured_models.add(cached_data["model_name"])
        return measured_models

    def evaluate_all_models(self, system_prompts: Dict[str, str] = None) -> None:
        """Оценивает все модели из конфига параллельно с использованием кэша"""
        if system_prompts is None:
            system_prompts = {}

        # Получаем список уже измеренных моделей
        measured_models = self._get_measured_models()

        # Получаем список всех моделей из конфига
        config_models = set(self.config["model_list"])

        # Находим новые модели
        new_models = config_models - measured_models

        if new_models:
            print(f"\nFound new models to evaluate: {', '.join(new_models)}")

        # Загружаем существующие кэши для всех моделей
        for model_name in config_models:
            if model_name in measured_models:
                # Загружаем кэш для существующей модели
                for cache_file in self.cache_dir.glob("*.json"):
                    with open(cache_file, "r") as f:
                        cached_data = json.load(f)
                        if cached_data["model_name"] == model_name:
                            key = f"{model_name}_{cached_data['timestamp']}"
                            self.results[key] = cached_data
                            break

        # Оцениваем только новые модели
        if new_models:
            uncached_args = [
                (model_name, system_prompts.get(model_name))
                for model_name in new_models
            ]

            print(f"\nEvaluating {len(uncached_args)} new models...")

            def handle_sigint(signum, frame):
                print(
                    "\nGracefully shutting down... Please wait for current evaluations to complete."
                )
                executor.shutdown(wait=True)
                sys.exit(0)

            original_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, handle_sigint)

            try:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Создаем futures для всех новых моделей
                    futures = [
                        executor.submit(self.evaluate_model_parallel, args)
                        for args in uncached_args
                    ]

                    # Создаем прогресс-бар с leave=True, чтобы он оставался после завершения
                    pbar = tqdm(
                        total=len(futures), desc="Evaluating new models", leave=True
                    )

                    # Обрабатываем каждый future по мере его завершения
                    # вместо итерации по futures, которая не отражает реальное завершение
                    completed = 0
                    while completed < len(futures):
                        # Проверяем статус каждого future
                        for i, future in enumerate(futures):
                            if future.done() and not hasattr(future, "_processed"):
                                try:
                                    result = future.result(timeout=1)
                                    if result:
                                        key = f"{result['model_name']}_{result['timestamp']}"
                                        self.results[key] = result
                                        # Сразу сохраняем результат в кэш
                                        self._save_to_cache(
                                            self._get_cache_key(
                                                result["model_name"],
                                                result.get("system_prompt"),
                                            ),
                                            result,
                                        )
                                    # Отмечаем future как обработанный
                                    setattr(future, "_processed", True)
                                    # Обновляем прогресс-бар только когда модель действительно завершена
                                    completed += 1
                                    pbar.update(1)
                                except TimeoutError:
                                    print(
                                        "\nWarning: Evaluation timed out for one of the models"
                                    )
                                except Exception as e:
                                    print(f"\nError during evaluation: {str(e)}")
                                    # Отмечаем future как обработанный даже при ошибке
                                    setattr(future, "_processed", True)
                                    completed += 1
                                    pbar.update(1)

                        # Не нагружаем CPU проверкой статуса
                        time.sleep(0.1)

                    # Закрываем прогресс-бар
                    pbar.close()

            finally:
                signal.signal(signal.SIGINT, original_sigint)
                self._save_results()
        else:
            print("\nNo new models to evaluate, using cached results")

        # Проверяем, что все модели из конфига присутствуют в результатах
        missing_models = config_models - set(
            result["model_name"] for result in self.results.values()
        )
        if missing_models:
            print(f"\nWarning: Missing results for models: {', '.join(missing_models)}")

        self._save_results()

    def generate_markdown(self) -> str:
        """Генерирует markdown с результатами"""
        md = "# Math Evaluation Leaderboard\n\n"
        md += f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Заголовок таблицы с добавлением колонок для Physics и общего скора
        md += "| Model | Combined Score | RussianMath Score | RussianPhysics Score | Tokens Used | System Prompt | Evaluation Time | Details |\n"
        md += "|-------|---------------|-------------------|----------------------|-------------|---------------|----------------|--------|\n"

        # Предварительно создаем комбинированные отчеты для всех моделей с обоими датасетами
        self._prepare_combined_reports()

        # Собираем данные по моделям для создания сводной таблицы
        model_data = {}

        for key, result in self.results.items():
            model_name = result["model_name"]

            if model_name not in model_data:
                model_data[model_name] = {
                    "combined": None,
                    "russianmath": None,
                    "physics": None,
                }

            # Определяем тип результата по датасету
            dataset = result.get("dataset")

            if dataset == "Combined":
                model_data[model_name]["combined"] = result
            elif dataset == "RussianMath" or (dataset is None):
                if (
                    not model_data[model_name]["russianmath"]
                    or result["score"] > model_data[model_name]["russianmath"]["score"]
                ):
                    model_data[model_name]["russianmath"] = result
            elif dataset == "RussianPhysics":
                if (
                    not model_data[model_name]["physics"]
                    or result["score"] > model_data[model_name]["physics"]["score"]
                ):
                    model_data[model_name]["physics"] = result

        # Сортируем модели по комбинированному скору, или при его отсутствии по Math скору
        def get_sort_score(model_name):
            data = model_data[model_name]
            if data["combined"]:
                return data["combined"]["score"]
            elif data["russianmath"]:
                return data["russianmath"]["score"]
            return 0

        sorted_models = sorted(model_data.keys(), key=get_sort_score, reverse=True)

        # Добавляем строки для каждой модели
        for model_name in sorted_models:
            data = model_data[model_name]

            # Пропускаем модели без результатов
            if not data["russianmath"] and not data["physics"]:
                continue

            # Получаем скоры
            combined_score = data["combined"]["score"] if data["combined"] else "-"
            rm_score = data["russianmath"]["score"] if data["russianmath"] else "-"
            physics_score = data["physics"]["score"] if data["physics"] else "-"

            # Получаем общее количество токенов
            total_tokens = 0
            if data["russianmath"]:
                total_tokens += data["russianmath"].get("total_tokens", 0)
            if data["physics"]:
                total_tokens += data["physics"].get("total_tokens", 0)

            # Получаем системный промпт
            system_prompt = None
            for result_type in ["russianmath", "physics", "combined"]:
                if data[result_type] and data[result_type].get("system_prompt"):
                    system_prompt = data[result_type]["system_prompt"]
                    break

            if system_prompt and len(system_prompt) > 30:
                system_prompt = system_prompt[:27] + "..."
            elif not system_prompt:
                system_prompt = "None"

            # Получаем суммарное время оценки
            eval_time = 0
            if data["russianmath"]:
                eval_time += data["russianmath"].get("evaluation_time", 0)
            if data["physics"]:
                eval_time += data["physics"].get("evaluation_time", 0)

            # Создаем ссылки на детали
            details = ""
            safe_model_name = model_name.replace("/", "_")

            # Проверяем наличие обоих отчетов и ищем комбинированный
            if data["russianmath"] and data["physics"]:
                math_timestamp = data["russianmath"]["timestamp"]
                physics_timestamp = data["physics"]["timestamp"]

                # Проверяем наличие комбинированного отчета
                combined_report = (
                    self.details_dir
                    / safe_model_name
                    / f"details_{math_timestamp}_combined.md"
                )
                if combined_report.exists():
                    details = f"[Combined](results/details/{safe_model_name}/details_{math_timestamp}_combined.md)"
                else:
                    # Пробуем с timestamp от физики
                    combined_report = (
                        self.details_dir
                        / safe_model_name
                        / f"details_{physics_timestamp}_combined.md"
                    )
                    if combined_report.exists():
                        details = f"[Combined](results/details/{safe_model_name}/details_{physics_timestamp}_combined.md)"
                    else:
                        # Если комбинированного отчета нет, используем отдельные ссылки
                        details = f"[Math](results/details/{safe_model_name}/details_{math_timestamp}.md) [Physics](results/details/{safe_model_name}/details_{physics_timestamp}_RussianPhysics.md)"
            else:
                # Если доступен только один тип отчета
                if data["russianmath"]:
                    timestamp = data["russianmath"]["timestamp"]
                    details = f"[Math](results/details/{safe_model_name}/details_{timestamp}.md)"
                elif data["physics"]:
                    timestamp = data["physics"]["timestamp"]
                    details = f"[Physics](results/details/{safe_model_name}/details_{timestamp}_RussianPhysics.md)"

            # Добавляем строку модели
            md += f"| {model_name} "
            md += f"| {combined_score if isinstance(combined_score, str) else f'{combined_score:.3f}'} "
            md += f"| {rm_score if isinstance(rm_score, str) else f'{rm_score:.3f}'} "
            md += f"| {physics_score if isinstance(physics_score, str) else f'{physics_score:.3f}'} "
            md += f"| {total_tokens} "
            md += f"| {system_prompt} "
            md += f"| {eval_time:.1f}s "
            md += f"| {details} |\n"

        # Сохраняем markdown в UTF-8
        with open(self.output_dir / "leaderboard.md", "w", encoding="utf-8") as f:
            f.write(md)

        return md

    def _prepare_combined_reports(self):
        """Подготавливает комбинированные отчеты для всех моделей с обоими датасетами"""
        # Получаем список всех моделей, у которых есть оба датасета
        models_with_both_datasets = {}

        # Сначала собираем данные по моделям и их датасетам
        for key, result in self.results.items():
            model_name = result["model_name"]
            dataset = result.get("dataset")
            timestamp = result.get("timestamp")

            if model_name not in models_with_both_datasets:
                models_with_both_datasets[model_name] = {"math": None, "physics": None}

            if dataset == "RussianMath" or dataset is None:
                if (
                    not models_with_both_datasets[model_name]["math"]
                    or timestamp > models_with_both_datasets[model_name]["math"]
                ):
                    models_with_both_datasets[model_name]["math"] = timestamp
            elif dataset == "RussianPhysics":
                if (
                    not models_with_both_datasets[model_name]["physics"]
                    or timestamp > models_with_both_datasets[model_name]["physics"]
                ):
                    models_with_both_datasets[model_name]["physics"] = timestamp

        # Теперь создаем комбинированные отчеты для моделей с обоими датасетами
        for model_name, timestamps in models_with_both_datasets.items():
            if timestamps["math"] and timestamps["physics"]:
                # Вызываем функцию создания комбинированного отчета
                self._combine_detailed_reports(
                    model_name=model_name,
                    timestamp_math=timestamps["math"],
                    timestamp_physics=timestamps["physics"],
                )

    def evaluate_math_demon_subsets(self):
        """Оценивает все подсеты из MathDemon_Dемидovich для всех моделей из конфига параллельно"""
        subsets = [
            "Approximation_by_Polynomials",
            "Continuous_Functions",
            "Convex_Functions",
            "Diﬀerentiation",
            "Improper_Integrals",
            "Infinite_Series",
            "Integration",
            "Sequences_and_Limits",
            "Series_of_Functions",
        ]

        print(f"\nEvaluating MathDemon_Dемидovich subsets ({len(subsets)} subsets)")

        # Словарь для хранения результатов всех моделей по всем подмножествам
        all_results = {model: {} for model in self.config["model_list"]}

        # Обрабатываем каждое подмножество последовательно
        for subset in subsets:
            print(f"\nEvaluating subset: {subset} for all models")

            # Для каждого подмножества обрабатываем все модели параллельно
            subset_results = self._evaluate_subset_parallel(subset)

            # Сохраняем результаты подмножества для каждой модели
            for model_name, result in subset_results.items():
                if result:  # Проверяем, что результат не None (на случай ошибки)
                    all_results[model_name][subset] = result

        # Вычисляем средние значения по всем подмножествам для каждой модели
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_name, subset_results in all_results.items():
            if subset_results:  # Если есть результаты для модели
                # Вычисляем среднее значение
                scores = [result["score"] for result in subset_results.values()]
                avg_score = sum(scores) / len(scores)

                # Суммируем токены и время выполнения
                total_tokens = sum(
                    result["total_tokens"] for result in subset_results.values()
                )
                total_time = sum(
                    result["evaluation_time"] for result in subset_results.values()
                )

                # Создаем запись с общим результатом для модели
                self.results[f"{model_name}_MathDemon_AllSubsets_{timestamp}"] = {
                    "model_name": model_name,
                    "score": avg_score,
                    "total_tokens": total_tokens,
                    "evaluation_time": total_time,
                    "system_prompt": self.config.get(model_name, {}).get(
                        "system_prompt"
                    ),
                    "timestamp": timestamp,
                    "dataset": "MathDemon_Dемидovich",
                    "subset": "AllSubsets",
                }

                print(
                    f"Model {model_name} average score across all MathDemon subsets: {avg_score:.3f}"
                )

        # Сохраняем все результаты
        self._save_results()

    def _evaluate_subset_parallel(self, subset_name):
        """Оценивает все модели на одном подмножестве MathDemon параллельно"""

        def evaluate_model_on_subset(model_name):
            """Оценивает одну модель на одном подмножестве MathDemon"""
            try:
                if self.config.get("debug"):
                    print(
                        f"Starting evaluation of model {model_name} on subset {subset_name}"
                    )

                # Создаем ключ кэша для этой оценки
                system_prompt = self.config.get(model_name, {}).get("system_prompt")
                cache_key = (
                    f"{self._get_cache_key(model_name, system_prompt)}_{subset_name}"
                )

                # Проверяем кэш
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    if self.config.get("debug"):
                        print(f"Using cached result for {model_name} on {subset_name}")
                    return cached_result

                # Создаем оценщик для текущего подмножества
                evaluator = MathDemonEval(
                    subset_name=subset_name,
                    num_examples=self.config.get("num_examples", None),
                    debug=self.config.get("debug", False),
                )

                # Устанавливаем equality_checker для проверки ответов
                evaluator.set_equality_checker(self.equality_checker)

                # Создаем временный конфиг для текущей модели
                temp_config = self.config.copy()
                temp_config["model_list"] = [model_name]

                temp_config_path = (
                    self.output_dir
                    / f"temp_config_mathdemon_{model_name}_{subset_name}.yaml"
                )
                with open(temp_config_path, "w", encoding="utf-8") as f:
                    yaml.dump(temp_config, f)

                sampler = OaiSampler(str(temp_config_path))

                start_time = time.time()
                results = evaluator(sampler)
                evaluation_time = time.time() - start_time

                # Сохраняем результаты для текущего подмножества
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Сохраняем детальные результаты в папку модели
                self._save_detailed_results(
                    model_name, results.results, timestamp, f"MathDemon_{subset_name}"
                )

                # Вычисляем общее количество токенов
                total_tokens = sum(
                    r.tokens for r in results.results if hasattr(r, "tokens")
                )

                # Создаем запись результата
                result_entry = {
                    "model_name": model_name,
                    "score": results.score,
                    "total_tokens": total_tokens,
                    "evaluation_time": evaluation_time,
                    "system_prompt": sampler.system_prompt,
                    "timestamp": timestamp,
                    "dataset": "MathDemon_Dемидovich",
                    "subset": subset_name,
                    "cache_key": cache_key,
                }

                # Добавляем в общие результаты
                self.results[f"{model_name}_MathDemon_{subset_name}_{timestamp}"] = (
                    result_entry
                )

                # Сохраняем в кэш
                self._save_to_cache(cache_key, result_entry)

                if not self.config.get("debug"):
                    # Выводим результат более кратко
                    print(
                        f"Model {model_name} on subset {subset_name}: {results.score:.3f}"
                    )
                else:
                    print(
                        f"Model {model_name} on subset {subset_name} score: {results.score:.3f}, tokens: {total_tokens}, time: {evaluation_time:.1f}s"
                    )

                # Удаляем временный конфиг
                temp_config_path.unlink(missing_ok=True)

                return result_entry

            except Exception as e:
                print(
                    f"Error evaluating subset {subset_name} for model {model_name}: {str(e)}"
                )
                if "temp_config_path" in locals():
                    temp_config_path.unlink(missing_ok=True)
                return None

        # Получаем список уже измеренных моделей
        measured_models = set()
        for key, result in self.results.items():
            if (
                result.get("dataset") == "MathDemon_Dемидovich"
                and result.get("subset") == subset_name
                and result.get("model_name") in self.config["model_list"]
            ):
                measured_models.add(result.get("model_name"))

        # Определяем модели для оценки (только новые)
        models_to_evaluate = set(self.config["model_list"]) - measured_models

        if not models_to_evaluate:
            print(
                f"All models already evaluated for subset {subset_name}, using cached results"
            )

            # Собираем результаты из кэша
            results = {}
            for model_name in self.config["model_list"]:
                for key, result in self.results.items():
                    if (
                        result.get("model_name") == model_name
                        and result.get("dataset") == "MathDemon_Dемидovich"
                        and result.get("subset") == subset_name
                    ):
                        results[model_name] = result
                        break

            return results

        print(f"Evaluating {len(models_to_evaluate)} models on subset {subset_name}")

        # Запускаем оценку всех моделей параллельно с помощью ThreadPoolExecutor
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Запускаем задачи для всех моделей
            future_to_model = {
                executor.submit(evaluate_model_on_subset, model_name): model_name
                for model_name in models_to_evaluate
            }

            # Создаем прогресс-бар
            with tqdm(
                total=len(future_to_model),
                desc=f"Evaluating models on {subset_name}",
                leave=True,
            ) as pbar:
                # Обрабатываем результаты по мере их готовности
                # вместо итерации по futures, которая не отражает реальное завершение
                completed = 0
                while completed < len(future_to_model):
                    # Проверяем статус каждого future
                    for future, model_name in list(future_to_model.items()):
                        if future.done() and not hasattr(future, "_processed"):
                            try:
                                result = future.result(timeout=1)
                                if result:
                                    results[model_name] = result
                                # Отмечаем future как обработанный
                                setattr(future, "_processed", True)
                                # Обновляем прогресс-бар только когда модель действительно завершена
                                completed += 1
                                pbar.update(1)
                            except TimeoutError:
                                print(
                                    f"\nWarning: Evaluation timed out for model {model_name}"
                                )
                            except Exception as e:
                                print(
                                    f"\nError during evaluation of model {model_name}: {str(e)}"
                                )
                                # Отмечаем future как обработанный даже при ошибке
                                setattr(future, "_processed", True)
                                completed += 1
                                pbar.update(1)

                    # Не нагружаем CPU проверкой статуса
                    time.sleep(0.1)

        # Добавляем существующие результаты из кэша
        for model_name in measured_models:
            for key, result in self.results.items():
                if (
                    result.get("model_name") == model_name
                    and result.get("dataset") == "MathDemon_Dемидovich"
                    and result.get("subset") == subset_name
                ):
                    results[model_name] = result
                    break

        return results

    def calculate_combined_scores(self):
        """Вычисляет комбинированный скор для каждой модели как полусумму результатов по обоим датасетам"""
        print("\nCalculating combined scores for models...")

        # Группируем результаты по моделям
        model_results = {}

        for key, result in self.results.items():
            model_name = result["model_name"]

            # Пропускаем объединенные результаты, чтобы не дублировать
            if key.endswith("_Combined"):
                continue

            if model_name not in model_results:
                model_results[model_name] = {
                    "RussianMath": None,
                    "RussianPhysics": None,
                }

            # Добавляем результаты в зависимости от датасета
            dataset = result.get("dataset", "RussianMath")

            if dataset == "RussianMath" and (
                model_results[model_name]["RussianMath"] is None
                or result["score"] > model_results[model_name]["RussianMath"]["score"]
            ):
                # Для RussianMath берем лучший результат
                model_results[model_name]["RussianMath"] = result

            elif dataset == "RussianPhysics" and (
                model_results[model_name]["RussianPhysics"] is None
                or result["score"]
                > model_results[model_name]["RussianPhysics"]["score"]
            ):
                # Для RussianPhysics берем лучший результат
                model_results[model_name]["RussianPhysics"] = result

        # Вычисляем комбинированные скоры для каждой модели
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_name, results in model_results.items():
            # Проверяем, что есть результаты по обоим датасетам
            if results["RussianMath"] and results["RussianPhysics"]:
                # Вычисляем полусумму скоров
                math_score = results["RussianMath"]["score"]
                physics_score = results["RussianPhysics"]["score"]
                combined_score = (math_score + physics_score) / 2.0

                # Получаем общее количество токенов и время оценки
                total_tokens = results["RussianMath"].get("total_tokens", 0) + results[
                    "RussianPhysics"
                ].get("total_tokens", 0)
                total_time = results["RussianMath"].get("evaluation_time", 0) + results[
                    "RussianPhysics"
                ].get("evaluation_time", 0)

                # Получаем system_prompt от любого из датасетов
                system_prompt = results["RussianMath"].get("system_prompt") or results[
                    "RussianPhysics"
                ].get("system_prompt")

                # Добавляем комбинированный результат
                self.results[f"{model_name}_Combined_{timestamp}"] = {
                    "model_name": model_name,
                    "score": combined_score,
                    "math_score": math_score,
                    "physics_score": physics_score,
                    "total_tokens": total_tokens,
                    "evaluation_time": total_time,
                    "system_prompt": system_prompt,
                    "timestamp": timestamp,
                    "dataset": "Combined",
                }

                print(
                    f"Model {model_name} combined score: {combined_score:.3f} (Math: {math_score:.3f}, Physics: {physics_score:.3f})"
                )

            elif results["RussianMath"]:
                print(
                    f"Warning: Model {model_name} has only RussianMath results, skipping combined score calculation"
                )

            elif results["RussianPhysics"]:
                print(
                    f"Warning: Model {model_name} has only RussianPhysics results, skipping combined score calculation"
                )

        # Сохраняем все результаты
        self._save_results()

    def evaluate_physics_models(self, system_prompts: Dict[str, str] = None) -> None:
        """Оценивает все модели на датасете RussianPhysics из конфига параллельно с использованием кэша"""
        if system_prompts is None:
            system_prompts = {}

        # Получаем список уже измеренных моделей
        measured_models = set()
        for key, result in self.results.items():
            if result.get("dataset") == "RussianPhysics":
                measured_models.add(result.get("model_name"))

        # Получаем список всех моделей из конфига
        config_models = set(self.config["model_list"])

        # Находим новые модели
        new_models = config_models - measured_models

        if new_models:
            print(
                f"\nFound new models to evaluate on RussianPhysics: {', '.join(new_models)}"
            )

        # Оцениваем только новые модели
        if new_models:
            uncached_args = [
                (model_name, system_prompts.get(model_name))
                for model_name in new_models
            ]

            print(f"\nEvaluating {len(uncached_args)} new models on RussianPhysics...")

            def handle_sigint(signum, frame):
                print(
                    "\nGracefully shutting down... Please wait for current evaluations to complete."
                )
                executor.shutdown(wait=True)
                sys.exit(0)

            original_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, handle_sigint)

            try:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Создаем futures для всех новых моделей
                    futures = [
                        executor.submit(self.evaluate_physics_model_parallel, args)
                        for args in uncached_args
                    ]

                    # Создаем прогресс-бар
                    pbar = tqdm(
                        total=len(futures),
                        desc="Evaluating new models on RussianPhysics",
                        leave=True,
                    )

                    # Обрабатываем каждый future по мере его завершения
                    completed = 0
                    while completed < len(futures):
                        # Проверяем статус каждого future
                        for i, future in enumerate(futures):
                            if future.done() and not hasattr(future, "_processed"):
                                try:
                                    result = future.result(timeout=1)
                                    if result:
                                        key = f"{result['model_name']}_{result['timestamp']}_physics"
                                        self.results[key] = result
                                        # Сразу сохраняем результат в кэш
                                        self._save_to_cache(
                                            f"{self._get_cache_key(result['model_name'], result.get('system_prompt'))}_physics",
                                            result,
                                        )
                                    # Отмечаем future как обработанный
                                    setattr(future, "_processed", True)
                                    # Обновляем прогресс-бар
                                    completed += 1
                                    pbar.update(1)
                                except TimeoutError:
                                    print(
                                        "\nWarning: Evaluation timed out for one of the models"
                                    )
                                except Exception as e:
                                    print(f"\nError during evaluation: {str(e)}")
                                    # Отмечаем future как обработанный даже при ошибке
                                    setattr(future, "_processed", True)
                                    completed += 1
                                    pbar.update(1)

                        # Не нагружаем CPU проверкой статуса
                        time.sleep(0.1)

                    # Закрываем прогресс-бар
                    pbar.close()

            finally:
                signal.signal(signal.SIGINT, original_sigint)
                self._save_results()
        else:
            print("\nNo new models to evaluate on RussianPhysics, using cached results")

        # Проверяем, что все модели из конфига присутствуют в результатах
        missing_models = config_models - set(
            result["model_name"]
            for result in self.results.values()
            if result.get("dataset") == "RussianPhysics"
        )
        if missing_models:
            print(
                f"\nWarning: Missing RussianPhysics results for models: {', '.join(missing_models)}"
            )

        self._save_results()

    def evaluate_physics_model(
        self, model_name: str, system_prompt: str = None
    ) -> Dict[str, Any]:
        """Оценивает одну модель на датасете RussianPhysics"""
        cache_key = f"{self._get_cache_key(model_name, system_prompt)}_physics"
        cached_result = self._get_cached_result(cache_key)

        if cached_result is not None:
            if self.config.get("debug"):
                print(f"\nUsing cached physics result for {model_name}")
            return cached_result

        if self.config.get("debug"):
            print(f"\nEvaluating {model_name} on RussianPhysics")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace("/", "_")

        # Создаем временный конфиг
        temp_config = self.config.copy()
        temp_config["model_list"] = [model_name]
        if system_prompt is not None:
            temp_config[model_name]["system_prompt"] = system_prompt

        temp_config_path = (
            self.output_dir / f"temp_config_physics_{safe_model_name}.yaml"
        )
        with open(temp_config_path, "w") as f:
            yaml.dump(temp_config, f)

        try:
            from src.mat_boy import RussianPhysicsEval

            sampler = OaiSampler(str(temp_config_path))
            evaluator = RussianPhysicsEval(
                equality_checker=self.equality_checker,
                num_examples=self.config.get("num_examples", None),
                debug=self.config.get("debug", False),
            )

            start_time = time.time()
            results = evaluator(sampler)
            evaluation_time = time.time() - start_time

            # Сохраняем детальные результаты
            self._save_detailed_results(
                model_name, results.results, timestamp, "RussianPhysics"
            )

            total_tokens = sum(
                r.tokens for r in results.results if hasattr(r, "tokens")
            )

            model_result = {
                "model_name": model_name,  # Сохраняем оригинальное имя
                "score": results.score,
                "total_tokens": total_tokens,
                "evaluation_time": evaluation_time,
                "system_prompt": system_prompt,
                "timestamp": timestamp,
                "cache_key": cache_key,
                "dataset": "RussianPhysics",
            }

            # Сохраняем в кэш
            self._save_to_cache(cache_key, model_result)

            # Используем оригинальное имя модели для ключа результатов
            self.results[f"{model_name}_{timestamp}_physics"] = model_result
            self._save_results()

            return model_result

        finally:
            temp_config_path.unlink(missing_ok=True)

    def evaluate_physics_model_parallel(self, args: tuple) -> Dict[str, Any]:
        """Оценивает одну модель на датасете RussianPhysics (для использования в ThreadPoolExecutor)"""
        model_name, system_prompt = args
        return self.evaluate_physics_model(model_name, system_prompt)

    def _combine_detailed_reports(
        self, model_name: str, timestamp_math: str = None, timestamp_physics: str = None
    ):
        """Combines math and physics detailed reports into a single MD file for the model"""
        # Create safe model name for directory
        safe_model_name = model_name.replace("/", "_")
        model_dir = self.details_dir / safe_model_name

        if not model_dir.exists():
            print(f"No detailed reports found for model {model_name}")
            return None

        # Если переданы конкретные timestamp'ы, используем их
        if timestamp_math and timestamp_physics:
            math_report = model_dir / f"details_{timestamp_math}.md"
            physics_report = (
                model_dir / f"details_{timestamp_physics}_RussianPhysics.md"
            )
        else:
            # Если timestamp'ы не указаны, найдем самые свежие отчеты
            math_reports = list(model_dir.glob("details_*.md"))
            physics_reports = list(model_dir.glob("details_*_RussianPhysics.md"))

            if not math_reports or not physics_reports:
                print(f"Missing either math or physics reports for model {model_name}")
                return None

            # Выберем самые свежие отчеты по дате модификации файла
            math_report = sorted(
                math_reports, key=lambda x: x.stat().st_mtime, reverse=True
            )[0]
            physics_report = sorted(
                physics_reports, key=lambda x: x.stat().st_mtime, reverse=True
            )[0]

            # Извлечем timestamp'ы из имен файлов
            timestamp_math = math_report.stem.split("_")[1]
            timestamp_physics = physics_report.stem.split("_")[1]

        # Проверяем существование файлов
        if not math_report.exists() or not physics_report.exists():
            print(f"Could not find both reports for model {model_name}")
            return None

        # Используем timestamp из математического отчета для комбинированного
        combined_report = model_dir / f"details_{timestamp_math}_combined.md"

        # Проверяем, существует ли уже комбинированный отчет
        if combined_report.exists():
            print(
                f"Combined report for {model_name} already exists at {combined_report}"
            )
            return combined_report

        # Read the content of both reports
        with open(math_report, "r", encoding="utf-8") as f:
            math_content = f.read()

        with open(physics_report, "r", encoding="utf-8") as f:
            physics_content = f.read()

        # Extract the content without headers to merge them
        math_lines = math_content.split("\n")
        physics_lines = physics_content.split("\n")

        # Find where the actual examples start in each file and where Summary section starts
        math_start = 0
        physics_start = 0
        math_summary_start = 0
        physics_summary_start = 0

        for i, line in enumerate(math_lines):
            if line.startswith("## Summary"):
                math_summary_start = i
            if line.startswith("## Example 1"):
                math_start = i
                break

        for i, line in enumerate(physics_lines):
            if line.startswith("## Summary"):
                physics_summary_start = i
            if line.startswith("## Example 1"):
                physics_start = i
                break

        # Create combined content
        # Start with the header but exclude Summary section
        combined_content = []
        for i in range(0, math_summary_start):
            if i == 0:
                # Replace the first line with a combined header
                combined_content.append(
                    f"# Detailed Results for {model_name} - Combined"
                )
            else:
                combined_content.append(math_lines[i])

        # Add combined summary section
        # Extract summary info from both reports
        math_summary_lines = [
            line
            for line in math_lines[math_summary_start:math_start]
            if "Score" in line
            or "examples" in line
            or "answers" in line
            or "Dataset" in line
        ]
        physics_summary_lines = [
            line
            for line in physics_lines[physics_summary_start:physics_start]
            if "Score" in line
            or "examples" in line
            or "answers" in line
            or "Dataset" in line
        ]

        # Extract scores from summaries
        math_score = next((line for line in math_summary_lines if "Score" in line), "")
        physics_score = next(
            (line for line in physics_summary_lines if "Score" in line), ""
        )

        try:
            # Extract numerical scores for average calculation
            math_score_value = float(math_score.split(":")[-1].strip())
            physics_score_value = float(physics_score.split(":")[-1].strip())
            combined_score = (math_score_value + physics_score_value) / 2
            combined_score_line = f"- **Combined Score**: {combined_score:.3f}"
        except (ValueError, IndexError):
            combined_score_line = "- **Combined Score**: N/A"

        combined_content.append("## Summary")
        combined_content.append("")
        combined_content.append(combined_score_line)
        combined_content.append("")
        combined_content.append("### Mathematics")
        for line in math_summary_lines:
            combined_content.append(line)
        combined_content.append("")
        combined_content.append("### Physics")
        for line in physics_summary_lines:
            combined_content.append(line)
        combined_content.append("")
        combined_content.append("---")
        combined_content.append("")

        # Add all math examples
        combined_content.append("## Mathematics Examples")
        combined_content.append("")
        combined_content.extend(math_lines[math_start:])

        # Add all physics examples
        combined_content.append("\n## Physics Examples")
        combined_content.append("")
        combined_content.extend(physics_lines[physics_start:])

        # Write the combined report
        with open(combined_report, "w", encoding="utf-8") as f:
            f.write("\n".join(combined_content))

        print(
            f"Successfully created combined report for {model_name} at {combined_report}"
        )
        return combined_report


def main():
    # Пример использования
    leaderboard = Leaderboard("configs/run.yaml")

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
