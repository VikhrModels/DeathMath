from typing import Dict, Union
import re
from src.types import SamplerBase
from fractions import Fraction
from decimal import Decimal, getcontext, ROUND_HALF_UP


class MathEqualityChecker(SamplerBase):
    """
    Класс для проверки равенства математических выражений.

    Выполняет нормализацию и сравнение математических выражений,
    учитывая различные форматы записи чисел и выражений.
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Инициализирует проверку равенства математических выражений.

        Args:
            debug: Режим отладки для вывода подробной информации
        """
        self.debug: bool = debug
        # Устанавливаем точность для decimal
        getcontext().prec = 3
        self.EPSILON: Decimal = Decimal("0.001")  # Точность сравнения

    def _pack_message(self, content: str, role: str = "user") -> Dict[str, str]:
        """
        Упаковывает содержимое в сообщение с указанной ролью.

        Args:
            content: Содержимое сообщения
            role: Роль отправителя (по умолчанию "user")

        Returns:
            Словарь, представляющий сообщение
        """
        return {"role": role, "content": content}

    def _normalize_answer(self, answer: Union[str, int, float]) -> str:
        """
        Нормализует ответ: преобразует в строку и оставляет только цифры,
        математические операторы и десятичный разделитель.

        Args:
            answer: Исходный ответ (строка или число)

        Returns:
            Нормализованная строка ответа
        """
        if not isinstance(answer, str):
            answer = str(answer)

        # Заменяем запятые на точки
        answer = answer.replace(",", ".")

        # Удаляем все пробелы
        answer = answer.replace(" ", "")

        # Удаляем символ доллара
        answer = answer.replace("$", "")

        # Заменяем русские буквы на английские (е -> e)
        answer = answer.replace("е", "e")

        # Оставляем только цифры, операторы и точку
        answer = re.sub(r"[^0-9\+\-\*/\(\)\.\,e]", "", answer)

        if self.debug:
            print(f"Normalized answer: '{answer}' from original: '{str(answer)}'")

        return answer

    def _evaluate_math_expression(self, expr: str) -> Union[float, str]:
        """
        Вычисляет математическое выражение.

        Безопасно вычисляет значение математического выражения,
        обрабатывая простые дроби и проверяя наличие недопустимых символов.

        Args:
            expr: Строка с математическим выражением

        Returns:
            Результат вычисления как число или исходная строка в случае ошибки

        Raises:
            ValueError: Если в выражении содержатся недопустимые символы
        """
        try:
            # Пробуем обработать дроби вида 1/10
            if "/" in expr and not any(op in expr for op in ["+", "-", "*"]):
                num, denom = expr.split("/")
                return float(Fraction(int(num), int(denom)))

            # Безопасное выполнение математического выражения
            allowed_chars = set("0123456789.+-*/() ")
            if not all(c in allowed_chars for c in expr):
                raise ValueError(f"Invalid characters in expression: {expr}")

            return float(eval(expr))

        except Exception as e:
            if self.debug:
                print(f"Error evaluating expression '{expr}': {str(e)}")
            return expr

    def __call__(
        self, expected: Union[str, int, float], actual: Union[str, int, float]
    ) -> bool:
        """
        Проверяет равенство математических ответов.

        Сравнивает два математических выражения или числа, нормализуя их
        и вычисляя значения с заданной точностью.

        Args:
            expected: Ожидаемый (правильный) ответ
            actual: Фактический ответ для проверки

        Returns:
            True если ответы эквивалентны, иначе False
        """
        if expected is None or actual is None:
            return False

        try:
            # Нормализуем ответы
            expected_norm = self._normalize_answer(expected)
            actual_norm = self._normalize_answer(actual)

            if self.debug:
                print(f"\nComparing: '{expected_norm}' with '{actual_norm}'")

            if expected_norm == actual_norm:
                return True

            # Пробуем сравнить как числа
            try:
                expected_val = self._evaluate_math_expression(expected_norm)
                actual_val = self._evaluate_math_expression(actual_norm)

                # Округляем до 3 знаков
                if isinstance(expected_val, (int, float, Fraction)):
                    expected_decimal = Decimal(str(float(expected_val))).quantize(
                        Decimal("0.001"), rounding=ROUND_HALF_UP
                    )
                else:
                    expected_decimal = Decimal(str(expected_val)).quantize(
                        Decimal("0.001"), rounding=ROUND_HALF_UP
                    )

                if isinstance(actual_val, (int, float, Fraction)):
                    actual_decimal = Decimal(str(float(actual_val))).quantize(
                        Decimal("0.001"), rounding=ROUND_HALF_UP
                    )
                else:
                    actual_decimal = Decimal(str(actual_val)).quantize(
                        Decimal("0.001"), rounding=ROUND_HALF_UP
                    )

                if self.debug:
                    print(f"Rounded values: {expected_decimal} vs {actual_decimal}")

                # Сравниваем округленные значения
                return abs(expected_decimal - actual_decimal) <= self.EPSILON

            except Exception as e:
                if self.debug:
                    print(f"Error during math evaluation: {str(e)}")
                return False

        except Exception as e:
            if self.debug:
                print(f"Error during equality check: {str(e)}")
            return False
