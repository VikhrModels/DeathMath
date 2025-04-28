import re
from fractions import Fraction
import sympy
from sympy.parsing.sympy_parser import parse_expr

# Если вдруг antlr4-python3-runtime не установлен, то не используем его
try:
    from sympy.parsing.latex import parse_latex  # type: ignore

    _HAS_PARSE_LATEX = True
except Exception:
    _HAS_PARSE_LATEX = False


class DoomSlayer:
    def __init__(self, EPS: float = 1e-2) -> None:
        self.EPS = EPS
        self.num_pattern = re.compile(r"-?\d+(?:[.,]\d+)?$")
        self.frac_pattern = re.compile(r"-?\d+\s*/\s*\d+$")
        # допускаем обрамления \( … \), $ … $, $$ … $$
        self.latex_frac_pattern = re.compile(
            r"(?:\\\(|\$\$?)?\s*\\frac\{(-?\d+)\}\{(\d+)\}\s*(?:\\\)|\$\$?)?"
        )
        self.minus_map = {"\u2212": "-", "\u2013": "-", "\u2014": "-"}

    # ──────────────────────────── service ────────────────────────────
    def _strip_delims(self, s: str) -> str:
        s = s.strip()
        if s.startswith("$$") and s.endswith("$$"):
            s = s[2:-2]
        elif s.startswith("$") and s.endswith("$"):
            s = s[1:-1]
        if s.startswith(r"\(") and s.endswith(r"\)"):
            s = s[2:-2]
        elif s.startswith(r"\[") and s.endswith(r"\]"):
            s = s[2:-2]
        return s.strip()

    def _normalize(self, s: str) -> str:
        for bad, good in self.minus_map.items():
            s = s.replace(bad, good)
        return s.strip()

    # ──────────────────────────── preprocessing ────────────────────────────
    def preprocess_answer(self, answer: str, hard: bool):
        answer = self._normalize(answer)
        answer = answer[:-1] if answer.endswith(".") else answer
        if not hard:
            return re.findall(r"-?\d+(?:[.,]\d+)?", answer)
        return [
            self._strip_delims(part).lower().replace("**", "^").strip()
            for part in answer.split(";")
        ]

    # ──────────────────────────── helpers ────────────────────────────
    def _compare_numeric(self, a: str, b: str) -> bool:
        """Абс- и относительная погрешность"""
        try:
            fa, fb = float(a.replace(",", ".")), float(b.replace(",", "."))
        except ValueError:
            return False
        diff = abs(fa - fb)
        return diff <= self.EPS or diff / (abs(fb) or 1) <= self.EPS

    def _compare_fraction(self, s1: str, s2: str) -> bool:
        def to_frac(s: str):
            s = self._strip_delims(s)
            if self.frac_pattern.fullmatch(s):
                num, den = map(int, s.split("/"))
                return Fraction(num, den)
            m = self.latex_frac_pattern.fullmatch(s)
            if m:
                num, den = map(int, m.groups())
                return Fraction(num, den)
            return None

        f1, f2 = to_frac(s1), to_frac(s2)
        if f1 is not None and f2 is not None:
            return abs(float(f1) - float(f2)) <= self.EPS
        return False

    def _to_expr(self, s: str):
        """Пытаемся превратить строку в sympy-выражение максимально надёжно"""
        # 1) голое число
        try:
            return sympy.Float(s.replace(",", "."))
        except Exception:
            pass
        # 2) обычная «python-математика»
        try:
            return parse_expr(s.replace("^", "**"), evaluate=True)
        except Exception:
            pass
        # 3) LaTeX (если библиотека доступна)
        if _HAS_PARSE_LATEX:
            try:
                return parse_latex(self._strip_delims(s))
            except Exception:
                pass
        return None

    def _expr_equal(self, e1, e2) -> bool:
        diff = sympy.simplify(e1 - e2)
        if diff == 0:
            return True  # точное равенство
        if diff.free_symbols:  # осталось x, y … — считаем неравными
            return False
        try:
            return abs(float(diff)) <= self.EPS
        except Exception:
            return False

    # ──────────────────────────── core ────────────────────────────
    def latex_equivalent(self, s1: str, s2: str) -> bool:
        p1, p2 = self.preprocess_answer(s1, True), self.preprocess_answer(s2, True)
        if len(p1) != len(p2):
            return False

        for a, b in zip(p1, p2):
            # быстрое сравнение чисел
            if self._compare_numeric(a, b):
                continue
            # пытаемся превратить в выражения
            e1, e2 = self._to_expr(a), self._to_expr(b)
            if e1 is None or e2 is None:
                if a != b:
                    return False
                continue
            if not self._expr_equal(e1, e2):
                return False
        return True

    # ──────────────────────────── public API ────────────────────────────
    def __call__(
        self, answer: str, predict: str
    ) -> bool:  # order: (правильный ответ, предикт)
        if not answer or not predict:
            return False
        answer, predict = self._normalize(answer), self._normalize(predict)

        if self._compare_fraction(answer, predict):
            return True

        if self.num_pattern.match(answer) and self.num_pattern.match(predict):
            if self._compare_numeric(answer, predict):
                return True
            # если равенство «почти» не прокатило – проверяем как выражения
            return self.latex_equivalent(predict, answer)

        return self.latex_equivalent(predict, answer)
