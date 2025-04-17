from dataclasses import dataclass, field
from typing import List, Dict, Optional, Protocol, Union, Any


@dataclass
class Message:
    """
    Структура сообщения в диалоге между пользователем и моделью.

    Attributes:
        role: Роль отправителя сообщения (system, user, assistant)
        content: Содержимое сообщения
        variant: Необязательный вариант сообщения
    """

    role: str
    content: str
    variant: Optional[str] = None


@dataclass
class SingleEvalResult:
    """
    Результат оценки одного примера.

    Attributes:
        score: Оценка примера (1.0 - правильно, 0.0 - неправильно, None - не применимо)
        metrics: Дополнительные метрики для данного примера
        html: HTML-представление результата для визуализации
        convo: Диалог с моделью в виде списка сообщений
        tokens: Количество использованных токенов
        correct_answer: Правильный ответ
        extracted_answer: Извлеченный ответ из ответа модели
    """

    score: Optional[float]
    metrics: Dict[str, float] = field(default_factory=dict)
    html: Optional[str] = None
    convo: Optional[List[Dict[str, str]]] = None
    tokens: int = 0
    correct_answer: Optional[str] = None
    extracted_answer: Optional[str] = None


@dataclass
class EvalResult:
    """
    Агрегированный результат оценки для множества примеров.

    Attributes:
        score: Общая оценка (обычно среднее значение оценок отдельных примеров)
        results: Список результатов для каждого примера
        metrics: Дополнительные метрики для всей оценки
        htmls: HTML-фрагменты для визуализации результатов
    """

    score: float
    results: List[SingleEvalResult]
    metrics: Dict[str, float] = field(default_factory=dict)
    htmls: List[str] = field(default_factory=list)


class SamplerBase(Protocol):
    """
    Протокол для взаимодействия с языковой моделью.

    Определяет интерфейс, который должны реализовать все классы,
    используемые для получения ответов от языковых моделей.
    """

    def _pack_message(self, content: str, role: str = "user") -> Dict[str, str]:
        """
        Упаковывает содержимое в сообщение с указанной ролью.

        Args:
            content: Содержимое сообщения
            role: Роль отправителя (по умолчанию "user")

        Returns:
            Словарь, представляющий сообщение

        Raises:
            NotImplementedError: Если метод не реализован в подклассе
        """
        raise NotImplementedError

    def __call__(
        self, messages: List[Dict[str, str]], return_metadata: bool = False
    ) -> Union[str, tuple[str, Dict[str, Any]]]:
        """
        Получает ответ от модели на заданные сообщения.

        Args:
            messages: Список сообщений для контекста
            return_metadata: Флаг для возврата метаданных вместе с ответом

        Returns:
            Строка ответа или кортеж (ответ, метаданные), если return_metadata=True

        Raises:
            NotImplementedError: Если метод не реализован в подклассе
        """
        raise NotImplementedError


class Eval(Protocol):
    """
    Протокол для оценки языковой модели.

    Определяет интерфейс для классов, которые оценивают
    производительность языковых моделей на различных задачах.
    """

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """
        Оценивает производительность модели на наборе примеров.

        Args:
            sampler: Экземпляр SamplerBase для взаимодействия с моделью

        Returns:
            Результат оценки

        Raises:
            NotImplementedError: Если метод не реализован в подклассе
        """
        raise NotImplementedError
