import re
import sympy
from sympy.parsing.latex import parse_latex


class DoomSlayer:
    def __init__(self, EPS=1e-2):
        self.EPS = EPS

    def preprocess_answer(self, answer: str, hard: bool) -> str:
        if not hard:
            return re.findall("[0-9.]+", answer)
        answer = answer.lower().replace("**", "^").split(";")
        return answer

    def __call__(self, predict: str, answer: str) -> bool:
        if not re.match("[0-9., ]+", answer) or not re.match("[0-9,. ]+", predict):
            return self.latex_equivalent(predict, answer)
        if (
            re.match("[0-9., ]+", answer)[0] == answer
            and re.match("[0-9,. ]+", predict)[0] == predict
        ):
            return self.simple_check(predict, answer)
        return self.latex_equivalent(predict, answer)

    def simple_check(self, predict: str, answer: str) -> bool:
        predict = self.preprocess_answer(predict, False)
        answer = self.preprocess_answer(answer, False)
        return "".join(answer).replace(",", ".") == "".join(predict).replace(",", ".")

    def latex_equivalent(self, latex_formula1: str, latex_formula2: str) -> bool:
        """
        Сравнивает две формулы в формате LaTeX через Sympy.
        Возвращает True, если формулы математически эквивалентны, иначе False.
        """
        latex_formula1 = self.preprocess_answer(latex_formula1, True)
        latex_formula2 = self.preprocess_answer(latex_formula2, True)

        results = [True for _ in range(len(latex_formula1))]
        for i in range(len(latex_formula1)):
            try:
                expr1 = parse_latex(latex_formula1[i])
                expr2 = parse_latex(latex_formula2[i])
                diff = sympy.simplify(expr1 - expr2)
                results[i] = diff <= self.EPS
            except Exception:
                try:
                    if latex_formula1[i] == latex_formula2[i]:
                        continue
                except:
                    pass
                results[i] = False
        return all(results)
