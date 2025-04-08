from typing import Optional, Any
import re
from src.types import SamplerBase
import numpy as np
from fractions import Fraction
from decimal import Decimal, getcontext, ROUND_HALF_UP

class MathEqualityChecker(SamplerBase):
    def _pack_message(self, content: str, role: str = "user") -> dict:
        return {"role": role, "content": content}
    
    def _normalize_answer(self, answer: str) -> str:
        """Нормализует ответ: удаляет все кроме цифр, операторов и точки"""
        if not isinstance(answer, str):
            answer = str(answer)
            
        # Заменяем запятые на точки
        answer = answer.replace(',', '.')
        
        # Удаляем все пробелы
        answer = answer.replace(' ', '')
        
        # Удаляем символ доллара
        answer = answer.replace('$', '')
        
        # Заменяем русские буквы на английские (е -> e)
        answer = answer.replace('е', 'e')
        
        # Оставляем только цифры, операторы и точку
        answer = re.sub(r'[^0-9\+\-\*/\(\)\.\,e]', '', answer)
        
        if self.debug:
            print(f"Normalized answer: '{answer}' from original: '{str(answer)}'")
        
        return answer

    def __call__(self, expected: str, actual: str) -> bool:
        """Проверяет равенство математических ответов"""
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
                    expected_decimal = Decimal(str(float(expected_val))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
                else:
                    expected_decimal = Decimal(str(expected_val)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
                    
                if isinstance(actual_val, (int, float, Fraction)):
                    actual_decimal = Decimal(str(float(actual_val))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
                else:
                    actual_decimal = Decimal(str(actual_val)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
                
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

    def _evaluate_math_expression(self, expr: str) -> Any:
        """Вычисляет математическое выражение"""
        try:
            # Пробуем обработать дроби вида 1/10
            if '/' in expr and not any(op in expr for op in ['+', '-', '*']):
                num, denom = expr.split('/')
                return float(Fraction(int(num), int(denom)))
            
            # Безопасное выполнение математического выражения
            allowed_chars = set('0123456789.+-*/() ')
            if not all(c in allowed_chars for c in expr):
                raise ValueError(f"Invalid characters in expression: {expr}")
                
            return float(eval(expr))
            
        except Exception as e:
            if self.debug:
                print(f"Error evaluating expression '{expr}': {str(e)}")
            return expr

    def __init__(self, debug: bool = False):
        self.debug = debug
        # Устанавливаем точность для decimal
        getcontext().prec = 3
        self.EPSILON = Decimal('0.001')  # Точность сравнения

    def _pack_message(self, content: str, role: str = "user") -> dict:
        return {"role": role, "content": content} 