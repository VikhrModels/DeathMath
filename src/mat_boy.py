import re
from typing import Literal
import pandas as pd
from datasets import load_dataset
from .common import check_equality, ANSWER_PATTERN, jinja_env, HTML_JINJA
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
        num_examples: int | None = None,
        n_repeats: int = 1,
        debug: bool = False,
    ):
        # Загружаем датасет
        dataset = load_dataset("Vikhrmodels/russian_math")
        examples = [
            {"task": row["task"], "Answer": row["short answer"]} 
            for row in dataset["train"]
        ]
        
        if num_examples:
            examples = examples[:num_examples]
        self.examples = examples * n_repeats
        self.equality_checker = equality_checker
        self.debug = debug

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            if self.debug:
                print(f"\nDebug: Processing example")
                print(f"Task: {row['task']}")
                print(f"Expected answer: {row['Answer']}")
            
            prompt_messages = [
                sampler._pack_message(content=QUERY_TEMPLATE_RU.format(**row), role="user")
            ]
            response_text = sampler(prompt_messages)
            
            # Ищем ответ после слов "Answer:" или "Ответ:"
            answer_pattern = r"(?:Answer|Ответ):\s*(.+)$"
            match = re.search(answer_pattern, response_text, re.MULTILINE)
            extracted_answer = match.group(1).strip() if match else None
            
            if self.debug:
                print(f"Extracted answer: {extracted_answer}")
            
            score = float(check_equality(self.equality_checker, str(row["Answer"]), extracted_answer))
            
            if self.debug:
                print(f"Score: {score}")
            
            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(html=html, score=score, convo=convo)

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)

