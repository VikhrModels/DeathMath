from __future__ import annotations

import re
from fractions import Fraction
from typing import Sequence

import sympy
from sympy.parsing.sympy_parser import parse_expr

# Если вдруг antlr4-python3-runtime не установлен, ­игнорируем latex-парсер
try:
    from sympy.parsing.latex import parse_latex  # type: ignore

    _HAS_PARSE_LATEX = True
except Exception:  # pragma: no cover
    _HAS_PARSE_LATEX = False


class DoomSlayer:
    """
    Проверка эквивалентности ответов (строка-к-строке).

    Главная идея ─ сначала пытаемся сравнить как числа,
    затем как дроби, затем как символьные выражения (sympy/LaTeX).
    """

    def __init__(self, EPS: float = 1e-2) -> None:
        self.EPS = EPS

        # ────── основные шаблоны ──────
        self.num_pattern = re.compile(r"-?\d+(?:[.,]\d+)?(?:[eE]-?\d+)?$")
        self.frac_pattern = re.compile(r"-?\d+\s*/\s*\d+$")
        # допускаем обрамление \( … \) , $ … $ , $$ … $$
        self.latex_frac_pattern = re.compile(
            r"(?:\\\(|\$\$?)?\s*\\frac\{(-?\d+)\}\{(\d+)\}\s*(?:\\\)|\$\$?)?"
        )
        # различные варианты "минуса"
        self.minus_map = {"\u2212": "-", "\u2013": "-", "\u2014": "-"}

    # ──────────────────────────── service ────────────────────────────
    def _strip_delims(self, s: str) -> str:
        """Убираем окружающие $ … $, $$ … $$, \( … \), \[ … \]"""
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
        """Замена длинных/узких минусов, обрезка пробелов"""
        for bad, good in self.minus_map.items():
            s = s.replace(bad, good)
        return s.strip()

    # ──────────────────────────── preprocessing ────────────────────────────
    def preprocess_answer(self, answer: str, hard: bool) -> list[str]:
        """
        * soft-режим (hard = False) нужен только для _compare_numeric —
          вытягиваем **все** числа в строке;
        * hard-режим - для символьного сравнения —
          разбиваем строку по «;» на отдельные выражения,
          убираем внешние $$, \( \) и приводим **^ → \*\*** .
        """
        answer = self._normalize(answer)
        answer = answer[:-1] if answer.endswith(".") else answer

        if not hard:
            return re.findall(r"-?\d+(?:[.,]\d+)?(?:[eE]-?\d+)?", answer)

        return [
            self._strip_delims(part).lower().replace("**", "^").strip()
            for part in answer.split(";")
        ]

    # ──────────────────────────── helpers ────────────────────────────
    def _compare_numeric(self, a: str, b: str) -> bool:
        """Абсолютная и относительная погрешность для чисел/научной нотации"""
        try:
            fa, fb = float(a.replace(",", ".")), float(b.replace(",", "."))
        except ValueError:
            return False
        diff = abs(fa - fb)
        return diff <= self.EPS or diff / (abs(fb) or 1.0) <= self.EPS

    def _compare_fraction(self, s1: str, s2: str) -> bool:
        """Сравнение обыкновенных дробей (в т. ч. LaTeX)"""
        def to_frac(s: str) -> Fraction | None:
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
        """
        Превращаем строку в sympy-объект (Float | Expr | Tuple | Matrix …).
        Пытаемся по очереди:
          1) число                                   → sympy.Float
          2) python-математика («2^3» → «2**3»)      → parse_expr
          3) LaTeX (если доступен parse_latex)
        """
        s_clean = s.replace(",", ".")  # «1,2» → «1.2»
        try:
            return sympy.Float(s_clean)
        except Exception:
            pass

        try:
            return parse_expr(s_clean.replace("^", "**"), evaluate=True)
        except Exception:
            pass

        if _HAS_PARSE_LATEX:
            try:
                return parse_latex(self._strip_delims(s_clean))
            except Exception:
                pass

        return None  # ничего не получилось

    # ──────────────────────────── expression equality ─────────────────────────
    def _iterable_equal(
        self,
        seq1: Sequence,
        seq2: Sequence,
    ) -> bool:
        """Рекурсивное покомпонентное сравнение кортежей / списков / матриц"""
        if len(seq1) != len(seq2):
            return False
        return all(self._expr_equal(a, b) for a, b in zip(seq1, seq2))

    def _expr_equal(self, e1, e2) -> bool:
        """
        Надёжное сравнение любых sympy-объектов:
        * скаляры (Float, Integer, Symbol …),
        * кортежи (sympy.Tuple / обычный tuple),
        * матрицы (sympy.MatrixBase).
        """
        # ─── кортежи / списки ───
        if isinstance(e1, (tuple, sympy.Tuple)) or isinstance(
            e2, (tuple, sympy.Tuple)
        ):
            if not isinstance(e1, (tuple, sympy.Tuple)) or not isinstance(
                e2, (tuple, sympy.Tuple)
            ):
                return False
            return self._iterable_equal(e1, e2)

        # ─── матрицы ───
        from sympy.matrices.matrices import MatrixBase  # локальный импорт → быстрее старт
        if isinstance(e1, MatrixBase) or isinstance(e2, MatrixBase):  # pragma: no branch
            if not (isinstance(e1, MatrixBase) and isinstance(e2, MatrixBase)):
                return False
            if e1.shape != e2.shape:
                return False
            return self._iterable_equal(tuple(e1), tuple(e2))

        # ─── обычные выражения ───
        try:
            diff = sympy.simplify(e1 - e2)
        except TypeError:
            # например, «tuple - Float» — значит несопоставимые типы
            return False

        if diff == 0:
            return True  # точное равенство

        if diff.free_symbols:  # остались x, y … → не удалось упростить
            return False

        try:
            return abs(float(diff)) <= self.EPS
        except Exception:  # pragma: no cover
            return False

    # ──────────────────────────── core ────────────────────────────
    def latex_equivalent(self, s1: str, s2: str) -> bool:
        """
        «Тяжёлое» сравнение:
            * разбиваем ответы по «;» (мультиответ),
            * сравниваем соответствующие пары.
        Порядок элементов **важен**.
        """
        p1 = self.preprocess_answer(s1, hard=True)
        p2 = self.preprocess_answer(s2, hard=True)
        if len(p1) != len(p2):
            return False

        for a, b in zip(p1, p2):
            # быстрое числовое сравнение
            if self._compare_numeric(a, b):
                continue

            # пробуем превратить в sympy-выражения
            e1, e2 = self._to_expr(a), self._to_expr(b)
            if e1 is None or e2 is None:
                # ничего не разобрали → сравниваем строково
                if a != b:
                    return False
                continue

            if not self._expr_equal(e1, e2):
                return False

        return True

    # ──────────────────────────── public API ────────────────────────────
    def __call__(self, answer: str, predict: str) -> bool:
        """
        Возможные случаи:
            * обыкновенные дроби                              «3/4»
            * числа/научная нотация                           «1e-3»
            * символика / LaTeX / python-выражения / мультиответ
        Порядок «правильный ответ, предсказание» сохраняётся
        для симметрии с тестами.
        """
        if not answer or not predict:
            return False

        answer = self._normalize(answer)
        predict = self._normalize(predict)

        # 1) дроби
        if self._compare_fraction(answer, predict):
            return True

        # 2) обе строки выглядят как «простое число»
        if self.num_pattern.fullmatch(answer) and self.num_pattern.fullmatch(predict):
            if self._compare_numeric(answer, predict):
                return True
            # если почти-равенство не прошло, проверяем как выражения
            return self.latex_equivalent(predict, answer)

        # 3) общий тяжёлый случай
        return self.latex_equivalent(predict, answer)
