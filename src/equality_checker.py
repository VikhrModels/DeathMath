from typing import Optional
import re
from src.types import SamplerBase

class MathEqualityChecker(SamplerBase):
    def _pack_message(self, content: str, role: str = "user") -> dict:
        return {"role": role, "content": content}
    
    def _normalize_answer(self, answer: str) -> Optional[str]:
        """Нормализует ответ для сравнения"""
        if answer is None:
            return None
            
        # Удаляем пробелы и переводим в нижний регистр
        answer = answer.strip().lower()
        
        # Удаляем все пробелы
        answer = re.sub(r'\s+', '', answer)
        
        # Заменяем запятые на точки для десятичных чисел
        answer = answer.replace(',', '.')
        
        # Удаляем единицы измерения и прочий текст, оставляем только числа
        answer = re.sub(r'[^0-9.-]', '', answer)
        
        return answer

    def __call__(self, correct_answer: str, predicted_answer: str) -> bool:
        """
        Проверяет равенство между правильным и предсказанным ответом
        
        Args:
            correct_answer: Правильный ответ из датасета
            predicted_answer: Предсказанный ответ от модели
            
        Returns:
            bool: True если ответы равны, False в противном случае
        """
        # Нормализуем оба ответа
        correct = self._normalize_answer(str(correct_answer))
        predicted = self._normalize_answer(predicted_answer)
        
        # Если какой-то из ответов None, возвращаем False
        if correct is None or predicted is None:
            return False
            
        # Проверяем равенство
        return correct == predicted 