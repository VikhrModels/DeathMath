import yaml
from src.equality_checker import MathEqualityChecker
from src.sampler import OaiSampler
from src.mat_boy import RussianMathEval

# Загружаем конфиг
with open('configs/run.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Создаем equality checker
equality_checker = MathEqualityChecker()

# Создаем сэмплер
sampler = OaiSampler('configs/run.yaml')

# Создаем эвалюатор
evaluator = RussianMathEval(
    equality_checker=equality_checker,
    num_examples=None,  # или None для всего датасета
    debug=config.get('debug', False)  # Используем значение из конфига
)

# Запускаем оценку
results = evaluator(sampler)
print(f"\nFinal score: {results.score}") 