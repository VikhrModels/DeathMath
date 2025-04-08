from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class Message:
    role: str
    content: str
    variant: Optional[str] = None

@dataclass
class SingleEvalResult:
    """Результат оценки одного примера"""
    score: float | None
    metrics: Dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: List[Dict[str, str]] | None = None  # sampled conversation
    tokens: int = 0
    correct_answer: str | None = None
    extracted_answer: str | None = None

@dataclass
class EvalResult:
    """Агрегированный результат оценки"""
    score: float
    results: List[SingleEvalResult]
    metrics: Dict[str, float] = field(default_factory=dict)
    htmls: List[str] = field(default_factory=list)

class SamplerBase:
    """Базовый класс для сэмплера"""
    def _pack_message(self, content: str, role: str = "user") -> Dict[str, str]:
        raise NotImplementedError

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError

class Eval:
    """Базовый класс для оценки"""
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError