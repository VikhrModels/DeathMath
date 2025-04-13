import re
from datasets import load_dataset
from .common import check_equality
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult
from . import common

QUERY_TEMPLATE_RU = """
Реши следующую математическую задачу пошагово. Последняя строка твоего ответа должна быть в формате Ответ: $ANSWER (без кавычек), где $ANSWER - это ответ на задачу.

{task}

Не забудь написать ответ в отдельной строке после "Ответ:", без использования команды \\boxed.
""".strip()


class RussianMathEval(Eval):
    def __init__(
        self,
        equality_checker: SamplerBase,
        num_examples: int | None = 5,
        n_repeats: int = 1,
        debug: bool = False,
    ):
        # Загружаем датасет
        dataset = load_dataset("Vikhrmodels/russian_math")
        examples = [
            {"task": row["task"], "Answer": row["short answer"]}
            for row in dataset["train"]
        ]

        # Ограничиваем количество примеров
        if num_examples and num_examples > 0:
            examples = examples[:num_examples]
        else:
            examples = examples[:5]  # По умолчанию берем 5 примеров

        self.examples = examples * n_repeats
        self.equality_checker = equality_checker
        self.debug = debug

        if self.debug:
            print(f"Loaded {len(self.examples)} examples for evaluation")

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            if self.debug:
                print("\nDebug: Processing example")
                print(f"Task: {row['task']}")
                print(f"Expected answer: {row['Answer']}")

            prompt_messages = [
                sampler._pack_message(
                    content=QUERY_TEMPLATE_RU.format(**row), role="user"
                )
            ]

            # Вызываем сэмплер и получаем ответ и метаданные (включая токены)
            response_text, metadata = sampler(prompt_messages, return_metadata=True)

            # Ищем ответ после слов "Answer:" или "Ответ:"
            answer_pattern = r"(?:Answer|Ответ):\s*(.+)$"
            match = re.search(answer_pattern, response_text, re.MULTILINE)
            extracted_answer = match.group(1).strip() if match else None

            if self.debug:
                print(f"Extracted answer: {extracted_answer}")

            score = float(
                check_equality(
                    self.equality_checker, str(row["Answer"]), extracted_answer
                )
            )

            if self.debug:
                print(f"Score: {score}")
                print(f"Tokens used: {metadata.get('total_tokens', 0)}")

            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
                tokens=metadata.get("total_tokens", 0),
            )

            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
                tokens=metadata.get("total_tokens", 0),
            )

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)


class MathDemonEval(Eval):
    def __init__(
        self, subset_name: str, num_examples: int | None = 1, debug: bool = False
    ):
        """Инициализация для оценки на подсетах MathDemon_Demidovich"""
        # Загружаем датасет с указанным подсетом
        dataset = load_dataset("Vikhrmodels/MathDemon_Demidovich", subset_name)
        examples = [
            {"task": row["translated_conditions"], "Answer": row["translated_answers"]}
            for row in dataset["train"]
        ]

        # Ограничиваем количество примеров
        if num_examples and num_examples > 0:
            examples = examples[:num_examples]
        else:
            examples = examples[:5]  # По умолчанию берём 5 примеров

        self.examples = examples
        self.debug = debug
        # Добавляем equality_checker для проверки ответов
        self.equality_checker = None

        if self.debug:
            print(f"Loaded {len(self.examples)} examples for subset {subset_name}")

    def set_equality_checker(self, equality_checker):
        """Устанавливает объект для проверки равенства ответов"""
        self.equality_checker = equality_checker

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """Выполняет оценку на основе предоставленного сэмплера"""

        def fn(row: dict):
            if self.debug:
                print("\nDebug: Processing example")
                print(f"Task: {row['task']}")
                print(f"Expected answer: {row['Answer']}")

            prompt_messages = [
                sampler._pack_message(
                    content=QUERY_TEMPLATE_RU.format(task=row["task"]), role="user"
                )
            ]

            # Вызываем сэмплер и получаем ответ и метаданные (включая токены)
            response_text, metadata = sampler(prompt_messages, return_metadata=True)

            # Ищем ответ после слов "Answer:" или "Ответ:"
            answer_pattern = r"(?:Answer|Ответ):\s*(.+)$"
            match = re.search(answer_pattern, response_text, re.MULTILINE)
            extracted_answer = match.group(1).strip() if match else None

            if self.debug:
                print(f"Extracted answer: {extracted_answer}")

            # Используем equality_checker, если он установлен
            score = 0.0
            if self.equality_checker:
                score = float(
                    check_equality(
                        self.equality_checker, str(row["Answer"]), extracted_answer
                    )
                )

            if self.debug:
                print(f"Score: {score}")
                print(f"Tokens used: {metadata.get('total_tokens', 0)}")

            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
                tokens=metadata.get("total_tokens", 0),
            )

            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
                tokens=metadata.get("total_tokens", 0),
            )

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)
