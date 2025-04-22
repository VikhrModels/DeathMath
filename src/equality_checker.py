from typing import Union
import re
from src.types import SamplerBase
import sympy
from sympy.parsing.latex import parse_latex


class DoomSlayer(SamplerBase):
    """
    Класс для проверки равенства математических выражений.

    Выполняет нормализацию и сравнение математических выражений,
    используя регулярные выражения и символьные вычисления через sympy.
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Инициализирует проверку равенства математических выражений.

        Args:
            debug: Режим отладки для вывода подробной информации
        """
        self.debug = debug

    def preprocess_answer(self, answer: str, hard: bool) -> str:
        """
        Предварительная обработка ответа перед сравнением.

        Args:
            answer: Исходный ответ
            hard: Режим строгого сравнения (True) или упрощенного (False)

        Returns:
            Предобработанный ответ в виде списка строк
        """
        if not hard:
            return sorted(re.findall("[0-9.]+", answer))
        answer = answer.lower()
        return sorted(re.findall(r"[0-9.,а-яa-z/\+\-]+", answer))

    def __call__(
        self, predict: Union[str, int, float], answer: Union[str, int, float]
    ) -> bool:
        """
        Проверяет равенство математических ответов.

        Args:
            predict: Предсказанный ответ для проверки
            answer: Ожидаемый (правильный) ответ

        Returns:
            True если ответы эквивалентны, иначе False
        """
        if predict is None or answer is None:
            return False

        predict = str(predict)
        answer = str(answer)

        if re.match("[0-9., ]+", answer) and re.match("[0-9,. ]+", predict):
            return self.simple_check(predict, answer)
        return self.latex_equivalent(predict, answer)

    def simple_check(self, predict: str, answer: str) -> bool:
        """
        Выполняет простую проверку числовых ответов.

        Args:
            predict: Предсказанный ответ
            answer: Ожидаемый ответ

        Returns:
            True если ответы совпадают, иначе False
        """
        predict = self.preprocess_answer(predict, False)
        answer = self.preprocess_answer(answer, False)
        if self.debug:
            print(answer, predict)
        return "".join(answer).replace(",", ".") == "".join(predict).replace(",", ".")

    def latex_equivalent(self, latex_formula1: str, latex_formula2: str) -> bool:
        """
        Сравнивает две формулы в формате LaTeX через Sympy.
        Возвращает True, если формулы математически эквивалентны, иначе False.

        Args:
            latex_formula1: Первая формула для сравнения
            latex_formula2: Вторая формула для сравнения

        Returns:
            True если формулы эквивалентны, иначе False
        """
        latex_formula1 = self.preprocess_answer(latex_formula1, True)
        latex_formula2 = self.preprocess_answer(latex_formula2, True)

        results = [True for _ in range(len(latex_formula1))]
        for i in range(len(latex_formula1)):
            try:
                expr1 = parse_latex(latex_formula1[i])
                expr2 = parse_latex(latex_formula2[i])
                diff = sympy.simplify(expr1 - expr2)
                results[i] = diff == 0
            except Exception as e:
                try:
                    if latex_formula1[i] == latex_formula2[i]:
                        continue
                except:
                    pass
                results[i] = False

                if self.debug:
                    print(f"Error during LaTeX comparison: {str(e)}")

        return all(results)
