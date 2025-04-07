import yaml
from typing import List, Dict, Any
import openai
from .types import SamplerBase

class OaiSampler(SamplerBase):
    def __init__(self, config_path: str):
        # Загружаем конфиг
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Получаем параметры для выбранной модели
        model_name = self.config['model_list'][0]  # Берем первую модель из списка
        self.model_config = self.config[model_name]
        
        # Получаем API ключ из конфига или переменной окружения
        api_key = self.config.get('api_key') or self.model_config['endpoints'][0].get('api_key')
        if not api_key:
            raise ValueError("API key not found in config")
        
        # Инициализируем клиент OpenAI
        endpoint = self.model_config['endpoints'][0]
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=endpoint.get('api_base', 'https://api.openai.com/v1')
        )
        
        self.model_name = self.model_config['model_name']
        self.temperature = self.config.get('temperature', 0.0)
        self.max_tokens = self.config.get('max_tokens', 2048)
        self.system_prompt = self.model_config.get('system_prompt', None)
        self.debug = self.config.get('debug', False)

    def _pack_message(self, content: str, role: str = "user") -> Dict[str, str]:
        """Упаковывает сообщение в формат для API"""
        return {"role": role, "content": content}

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        """Отправляет запрос к API и возвращает ответ"""
        if self.debug:
            print("\nDebug: Sending request to API")
            print(f"Model: {self.model_name}")
            print("Messages:")
            for msg in messages:
                print(f"{msg['role']}: {msg['content'][:100]}...")
        
        # Добавляем system prompt если он есть
        if self.system_prompt:
            messages = [
                self._pack_message(content=self.system_prompt, role="system")
            ] + messages
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        if self.debug:
            print("\nDebug: Received response")
            print(f"Response: {response.choices[0].message.content[:100]}...")
        
        return response.choices[0].message.content 