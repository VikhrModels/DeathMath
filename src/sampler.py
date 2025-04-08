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
        self.model_config = self.config.get(model_name, {})
        
        # Получаем API ключ из конфига
        self.api_key = self.config.get('api_key')  # Сначала проверяем общий ключ
        if not self.api_key and 'endpoints' in self.model_config:
            # Если нет общего ключа, ищем в endpoints модели
            self.api_key = self.model_config['endpoints'][0].get('api_key')
        
        if not self.api_key:
            raise ValueError(f"API key not found in config for model {model_name}")
            
        # Получаем base_url
        self.base_url = None
        if 'endpoints' in self.model_config:
            self.base_url = self.model_config['endpoints'][0].get('api_base')
        
        # Инициализируем клиент OpenAI
        if self.base_url:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            self.client = openai.OpenAI(api_key=self.api_key)
        
        self.model_name = self.model_config.get('model_name', model_name)
        self.temperature = self.config.get('temperature', 0.0)
        self.max_tokens = self.config.get('max_tokens', 2048)
        self.system_prompt = self.model_config.get('system_prompt', None)
        self.debug = self.config.get('debug', False)

        if self.debug:
            print(f"\nDebug: Initialized OaiSampler")
            print(f"Model: {self.model_name}")
            print(f"Base URL: {self.base_url}")
            print(f"API Key: {self.api_key[:8]}...")

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
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            if self.debug:
                print("\nDebug: Received response")
                print(f"Response cutted to 100 chars: {response.choices[0].message.content[:100]}...")
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = (
                f"\nError during API call:"
                f"\nModel: {self.model_name}"
                f"\nBase URL: {self.base_url}"
                f"\nAPI Key (first 8 chars): {self.api_key[:8]}..."
                f"\nError: {str(e)}"
            )
            print(error_msg)
            raise Exception(error_msg) from e 