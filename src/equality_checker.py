import numpy as np
import pandas as pd
import json
import sympy
from sympy.parsing.latex import parse_latex
import os
import re
from fractions import Fraction


class DoomSlayer:
    def __init__(self, EPS=1e-2):
        self.EPS = EPS
        self.num_pattern = re.compile(r"-?\d+(?:[.,]\d+)?$")
        self.frac_pattern = re.compile(r"-?\d+\s*/\s*\d+$")
        self.latex_frac_pattern = re.compile(r"\\frac\{(-?\d+)\}\{(\d+)\}")
        self.minus_map = {"\u2212": "-", "\u2013": "-", "\u2014": "-"}  # ADDED

    def _normalize(self, s: str) -> str:
        for uni, ascii_minus in self.minus_map.items():  # ADDED
            s = s.replace(uni, ascii_minus)  # ADDED
        return s.strip()  # ADDED

    def preprocess_answer(self, answer: str, hard: bool):
        answer = self._normalize(answer)  # ADDED
        answer = answer[:-1] if answer.endswith(".") else answer
        if not hard:
            return re.findall(r"-?\d+(?:[.,]\d+)?", answer)
        return answer.lower().replace("**", "^").split(";")

    def __call__(self, answer: str, predict: str) -> bool:
        if not answer or not predict:
            return False
        answer = self._normalize(answer)  # ADDED
        predict = self._normalize(predict)  # ADDED

        if self._compare_fraction(answer, predict):
            return True

        if self.num_pattern.match(answer) and self.num_pattern.match(predict):
            return self.simple_check(predict, answer) or self.latex_equivalent(
                predict, answer
            )

        return self.latex_equivalent(predict, answer)

    def simple_check(self, predict: str, answer: str) -> bool:
        p = self.preprocess_answer(predict, False)
        a = self.preprocess_answer(answer, False)
        return "".join(a).replace(",", ".") == "".join(p).replace(",", ".")

    def _compare_fraction(self, s1: str, s2: str) -> bool:
        def to_frac(s):
            s = s.strip()
            m = self.frac_pattern.fullmatch(s)
            if m:
                num, den = map(int, s.split("/"))
                return Fraction(num, den)
            m2 = self.latex_frac_pattern.fullmatch(s)
            if m2:
                num, den = map(int, m2.groups())
                return Fraction(num, den)
            return None

        f1 = to_frac(s1)
        f2 = to_frac(s2)
        if f1 is not None and f2 is not None:
            return abs(float(f1) - float(f2)) <= self.EPS
        return False

    def latex_equivalent(self, latex1: str, latex2: str) -> bool:
        parts1 = self.preprocess_answer(latex1, True)
        parts2 = self.preprocess_answer(latex2, True)
        if len(parts1) != len(parts2):
            return False

        for a, b in zip(parts1, parts2):
            try:
                e1 = parse_latex(a)
                e2 = parse_latex(b)
                diff = sympy.simplify(abs(e1 - e2))
                try:
                    diff_rel = sympy.simplify(abs(e1 - e2) / abs(e2))
                    diff = min(diff, diff_rel)
                except Exception:
                    pass
                if diff > self.EPS:
                    return False
            except Exception:
                if a != b:
                    return False

        return True
