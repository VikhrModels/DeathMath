# https://github.com/openai/simple-evals/blob/main/common.py
# all creds to openai

from typing import Any, List, Callable, Dict, TypeVar, Union, Optional


import io
import jinja2
import numpy as np
import requests
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

from .types import EvalResult, SingleEvalResult

T = TypeVar("T")
R = TypeVar("R")

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN = r"(?:Answer|Ответ):\s*(.+)$"
MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    "(?i){}[ \t]*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"
)
# All the different ways "Answer" is written in different languages
MULTILINGUAL_ANSWER_REGEXES = [
    r"Answer\s*:",
    r"Answer\s*:​​​​​​",  # Korean invisible character
    r"উত্তর\s*:",
    r"उत्तर\s*:",
    r"উত্তরঃ",
    r"উত্তর\s*:",
    r"Antwort\s*:",
    r"답변\s*:",
    r"정답\s*:",
    r"답\s*:",
    r"答案\s*：",
    r"答案\s*:",
    r"答\s*：",
    r"答\s*:",
    r"答复\s*：",
    r"答曰\s*：",
    r"الإجابة:",
    r"الجواب:",
    r"إجابة:",
    r"الإجابة النهائية:",
    r"الإجابة الصحيحة:",
    r"الإجابة الصحيحة هي:",
    r"الإجابة هي:",
    r"الجواب النهائي:",
    r"Respuesta\s*:",
    r"Risposta\s*:",
    r"答え\s*:",
    r"答え\s*：",
    r"回答\s*:",
    r"回答\s*：",
    r"解答\s*:",
    r"Jawaban\s*:",
    r"Réponse\s*:",
    r"Resposta\s*:",
    r"Jibu\s*:",
    r"Idahun\s*:",
    r"Ìdáhùn\s*:",
    r"Idáhùn\s*:",
    r"Àmọ̀nà\s*:",
    r"Àdáhùn\s*:",
    r"Ànúgọ\s*:",
    r"Àṣàyàn\s*:",
]


EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()


HTML_JINJA = """
<div style="border: 1px solid #ccc; padding: 10px; margin: 10px 0;">
    <div style="margin-bottom: 10px;">
        <strong>Prompt:</strong><br>
        {% for msg in prompt_messages %}
            <div>{{ msg.content }}</div>
        {% endfor %}
    </div>
    <div style="margin-bottom: 10px;">
        <strong>Response:</strong><br>
        {{ next_message.content }}
    </div>
    <div>
        <strong>Score:</strong> {{ score }}<br>
        <strong>Correct answer:</strong> {{ correct_answer }}<br>
        <strong>Extracted answer:</strong> {{ extracted_answer }}
    </div>
</div>
"""


def format_multichoice_question(row: Dict[str, str]) -> str:
    """
    Форматирует вопрос с множественным выбором из данных строки.

    Args:
        row: Словарь с данными вопроса, содержащий ключи Question, A, B, C, D

    Returns:
        Отформатированный текст вопроса
    """
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def check_equality(equality_checker: Any, correct: str, predicted: str) -> bool:
    """
    Проверяет равенство ответов с помощью equality checker.

    Args:
        equality_checker: Объект для проверки равенства математических выражений
        correct: Правильный ответ
        predicted: Предсказанный ответ

    Returns:
        True если ответы эквивалентны, иначе False
    """
    return equality_checker(correct, predicted)


def _compute_stat(values: List[float], stat: str) -> float:
    """
    Вычисляет статистические показатели для списка значений.

    Args:
        values: Список числовых значений
        stat: Тип статистики ('mean', 'std', 'min', 'max')

    Returns:
        Вычисленное статистическое значение

    Raises:
        ValueError: Если указан неизвестный тип статистики
    """
    if stat == "mean":
        return np.mean(values)
    elif stat == "std":
        return np.std(values)
    elif stat == "min":
        return np.min(values)
    elif stat == "max":
        return np.max(values)
    else:
        raise ValueError(f"Unknown {stat =}")


def aggregate_results(results: List[SingleEvalResult]) -> EvalResult:
    """
    Агрегирует список результатов оценки в один объект EvalResult.

    Args:
        results: Список результатов оценки для отдельных примеров

    Returns:
        Агрегированный результат оценки
    """
    total_score = sum(r.score for r in results if r.score is not None)
    count = sum(1 for r in results if r.score is not None)

    return EvalResult(score=total_score / count if count > 0 else 0.0, results=results)


def map_with_progress(
    fn: Callable[[T], R],
    items: List[T],
    max_workers: int = 4,
    model_name: Optional[str] = None,
) -> List[R]:
    """
    Параллельно применяет функцию к элементам списка с отображением прогресса.

    Args:
        fn: Функция для применения к каждому элементу
        items: Список элементов для обработки
        max_workers: Максимальное количество параллельных потоков
        model_name: (Опционально) Имя модели для отображения в прогресс-баре


    Returns:
        Список результатов применения функции
    """

    desc = (
        f"Processing examples for {model_name}" if model_name else "Processing examples"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(fn, items),
                total=len(items),
                desc=desc,  # Используем обновленное описание
                leave=False,
            )
        )
    return results


jinja_env = jinja2.Environment()
_message_template = """
<div class="message {{ role }}">
    <div class="role">
    {{ role }}
    {% if variant %}<span class="variant">({{ variant }})</span>{% endif %}
    </div>
    <div class="content">
    <pre>{{ content }}</pre>
    </div>
</div>
"""


def message_to_html(message: Dict[str, str]) -> str:
    """
    Генерирует HTML-фрагмент для сообщения.

    Args:
        message: Словарь с ключами 'role', 'content' и опционально 'variant'

    Returns:
        HTML-представление сообщения
    """
    return jinja_env.from_string(_message_template).render(
        role=message["role"],
        content=message["content"],
        variant=message.get("variant", None),
    )


jinja_env.globals["message_to_html"] = message_to_html


_report_template = """<!DOCTYPE html>
<html>
    <head>
        <style>
            .message {
                padding: 8px 16px;
                margin-bottom: 8px;
                border-radius: 4px;
            }
            .message.user {
                background-color: #B2DFDB;
                color: #00695C;
            }
            .message.assistant {
                background-color: #B39DDB;
                color: #4527A0;
            }
            .message.system {
                background-color: #EEEEEE;
                color: #212121;
            }
            .role {
                font-weight: bold;
                margin-bottom: 4px;
            }
            .variant {
                color: #795548;
            }
            table, th, td {
                border: 1px solid black;
            }
            pre {
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
    {% if metrics %}
    <h1>Metrics</h1>
    <table>
    <tr>
        <th>Metric</th>
        <th>Value</th>
    </tr>
    <tr>
        <td><b>Score</b></td>
        <td>{{ score | float | round(3) }}</td>
    </tr>
    {% for name, value in metrics.items() %}
    <tr>
        <td>{{ name }}</td>
        <td>{{ value }}</td>
    </tr>
    {% endfor %}
    </table>
    {% endif %}
    <h1>Examples</h1>
    {% for html in htmls %}
    {{ html | safe }}
    <hr>
    {% endfor %}
    </body>
</html>
"""


def make_report(eval_result: EvalResult) -> str:
    """
    Создает автономный HTML-отчет из объекта EvalResult.

    Args:
        eval_result: Результат оценки

    Returns:
        HTML-строка с отчетом
    """
    return jinja_env.from_string(_report_template).render(
        score=eval_result.score,
        metrics=eval_result.metrics,
        htmls=eval_result.htmls,
    )


def make_report_from_example_htmls(htmls: List[str]) -> str:
    """
    Создает автономный HTML-отчет из списка HTML-фрагментов примеров.

    Args:
        htmls: Список HTML-фрагментов

    Returns:
        HTML-строка с отчетом
    """
    return jinja_env.from_string(_report_template).render(
        score=None, metrics={}, htmls=htmls
    )


def normalize_response(response: str) -> str:
    """
    Нормализует ответ, удаляя форматирование markdown и LaTeX.

    Убирает специальные символы и форматирование, которые могут
    помешать точному сопоставлению ответов.

    Args:
        response: Исходный ответ

    Returns:
        Нормализованный ответ
    """
    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


def normalize_extracted_answer(extracted_answer: str) -> str:
    """
    Нормализует извлеченный ответ, заменяя буквы на латинские эквиваленты.

    Заменяет многоязычные символы на стандартные латинские буквы A-D
    для вопросов с множественным выбором.

    Args:
        extracted_answer: Исходный ответ

    Returns:
        Нормализованный ответ
    """
    return (
        # In arabic these are the letters used for A-D in multiple choice questions
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        # In Bengali these are the letters used for A-D in multiple choice questions
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple choice questions
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )


def url_to_fileobj(url: str, binary: bool = False) -> Union[io.StringIO, io.BytesIO]:
    """
    Получает содержимое URL и возвращает его как файловый объект.

    Args:
        url: URL для загрузки
        binary: Если True, возвращает бинарный файловый объект

    Returns:
        StringIO или BytesIO с содержимым URL

    Raises:
        requests.exceptions.RequestException: Если загрузка не удалась
    """
    response = requests.get(url)
    response.raise_for_status()
    return io.BytesIO(response.content) if binary else io.StringIO(response.text)
