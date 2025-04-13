import yaml
from typing import List, Dict
import openai
import time
from .types import SamplerBase
from gigachat import GigaChat
from gigachat.models import Chat, Messages

# Maximum number of API retries and sleep time between retries
API_MAX_RETRY = 3
API_RETRY_SLEEP = 2
API_ERROR_OUTPUT = "Error during API call. Please try again."


class OaiSampler(SamplerBase):
    def __init__(self, config_path: str):
        # Загружаем конфиг
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Получаем параметры для выбранной модели
        model_name = self.config["model_list"][0]  # Берем первую модель из списка
        self.model_config = self.config.get(model_name, {})

        # Определяем тип API
        self.api_type = self.model_config.get("api_type", "openai")

        # Получаем параметры из endpoints
        if "endpoints" in self.model_config:
            endpoint = self.model_config["endpoints"][0]

            # Унифицированное получение API ключа или credentials
            self.api_key = endpoint.get("api_key", self.config.get("api_key"))
            self.credentials = endpoint.get("credentials")

            # Унифицированное получение base_url
            self.base_url = endpoint.get("api_base", endpoint.get("base_url"))

            # Дополнительные параметры для GigaChat
            self.scope = endpoint.get("scope", "GIGACHAT_API_CORP")
            self.profanity_check = endpoint.get("profanity_check", True)
            self.timeout = endpoint.get("timeout", 60.0)
        else:
            self.api_key = self.config.get("api_key")
            self.credentials = None
            self.base_url = None
            self.scope = "GIGACHAT_API_CORP"
            self.profanity_check = True
            self.timeout = 60.0

        # Проверка наличия необходимых учетных данных
        if self.api_type == "openai" and not self.api_key:
            raise ValueError(f"API key not found in config for model {model_name}")
        elif self.api_type == "gigachat" and not self.credentials:
            raise ValueError(f"Credentials not found in config for model {model_name}")

        # Инициализируем клиент OpenAI если нужно
        self.client = None
        if self.api_type == "openai":
            if self.base_url:
                self.client = openai.OpenAI(
                    api_key=self.api_key, base_url=self.base_url
                )
            else:
                self.client = openai.OpenAI(api_key=self.api_key)

        self.model_name = self.model_config.get("model_name", model_name)
        self.temperature = self.config.get("temperature", 0.0)

        # Получаем max_tokens из настроек конкретной модели, если он там есть
        # Иначе используем общее значение из конфига или значение по умолчанию
        self.max_tokens = self.model_config.get(
            "max_tokens", self.config.get("max_tokens", 2048)
        )

        self.system_prompt = self.model_config.get("system_prompt", None)
        self.debug = self.config.get("debug", False)

        if self.debug:
            print("\nDebug: Initialized OaiSampler")
            print(f"Model: {self.model_name}")
            print(f"API Type: {self.api_type}")
            print(f"Base URL: {self.base_url}")
            if self.api_key:
                print(f"API Key: {self.api_key[:8]}...")
            elif self.credentials:
                print(f"Using credentials for {self.api_type}")

    def _pack_message(self, content: str, role: str = "user") -> Dict[str, str]:
        """Упаковывает сообщение в формат для API"""
        return {"role": role, "content": content}

    def chat_completion_gigachat(self, model, messages, temperature, max_tokens):
        """Обработка запроса к GigaChat API"""

        # Создаем api_dict для GigaChat из унифицированных параметров
        api_dict = {
            "credentials": self.credentials,
            "base_url": self.base_url,
            "scope": self.scope,
            "profanity_check": self.profanity_check,
            "timeout": self.timeout,
        }

        client = GigaChat(model=model, verify_ssl_certs=False, **api_dict)

        # Настраиваем параметры для GigaChat
        top_p = 1
        if temperature == 0:
            temperature = 1
            top_p = 0

        # Преобразуем сообщения в формат GigaChat
        giga_messages = [Messages.parse_obj(m) for m in messages]
        chat = Chat(
            messages=giga_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        output = API_ERROR_OUTPUT
        metadata = {"total_tokens": 0}

        for _ in range(API_MAX_RETRY):
            try:
                response = client.chat(chat)
                output = response.choices[0].message.content

                # Извлекаем информацию о токенах
                if hasattr(response, "usage") and response.usage:
                    metadata["prompt_tokens"] = getattr(
                        response.usage, "prompt_tokens", 0
                    )
                    metadata["completion_tokens"] = getattr(
                        response.usage, "completion_tokens", 0
                    )
                    metadata["total_tokens"] = getattr(
                        response.usage, "total_tokens", 0
                    )

                if self.debug:
                    print(f"Tokens used: {metadata['total_tokens']}")

                break
            except Exception as e:
                if self.debug:
                    print(f"GigaChat API error: {type(e)} {str(e)}")
                time.sleep(API_RETRY_SLEEP)

        return output, metadata

    def __call__(self, messages: List[Dict[str, str]], return_metadata: bool = False):
        """Отправляет запрос к API и возвращает ответ

        Args:
            messages: Список сообщений для отправки
            return_metadata: Если True, возвращает также метаданные (например, использование токенов)

        Returns:
            str или tuple: Если return_metadata=False, возвращает только текст ответа.
                          Если return_metadata=True, возвращает кортеж (текст_ответа, метаданные)
        """
        if self.debug:
            print("\nDebug: Sending request to API")
            print(f"Model: {self.model_name}")
            print(f"API Type: {self.api_type}")
            print("Messages:")
            for msg in messages:
                print(f"{msg['role']}: {msg['content'][:100]}...")

        # Добавляем system prompt если он есть
        if self.system_prompt:
            messages = [
                self._pack_message(content=self.system_prompt, role="system")
            ] + messages

        try:
            # Обработка в зависимости от типа API
            if self.api_type == "gigachat":
                result, metadata = self.chat_completion_gigachat(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                if return_metadata:
                    return result, metadata
                return result

            else:  # openai API по умолчанию
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                if self.debug:
                    print("\nDebug: Received response")
                    print(f"Response type: {type(response)}")

                # Инициализируем метаданные
                metadata = {"total_tokens": 0}

                # Извлекаем информацию о токенах из разных типов ответов
                if hasattr(response, "usage"):
                    metadata["prompt_tokens"] = getattr(
                        response.usage, "prompt_tokens", 0
                    )
                    metadata["completion_tokens"] = getattr(
                        response.usage, "completion_tokens", 0
                    )
                    metadata["total_tokens"] = getattr(
                        response.usage, "total_tokens", 0
                    )
                elif isinstance(response, dict) and "usage" in response:
                    metadata["prompt_tokens"] = response["usage"].get(
                        "prompt_tokens", 0
                    )
                    metadata["completion_tokens"] = response["usage"].get(
                        "completion_tokens", 0
                    )
                    metadata["total_tokens"] = response["usage"].get("total_tokens", 0)

                if self.debug and metadata["total_tokens"] > 0:
                    print(f"Tokens used: {metadata['total_tokens']}")

                try:
                    # Стандартный путь для OpenAI API
                    if hasattr(response, "choices") and len(response.choices) > 0:
                        if hasattr(response.choices[0], "message") and hasattr(
                            response.choices[0].message, "content"
                        ):
                            result = response.choices[0].message.content
                            if self.debug:
                                print(
                                    f"Response content (first 100 chars): {result[:100]}..."
                                )

                            if return_metadata:
                                return result, metadata
                            return result

                    # Путь для словарного формата (некоторые API, включая OpenRouter)
                    if isinstance(response, dict) and "choices" in response:
                        if len(response["choices"]) > 0:
                            if (
                                "message" in response["choices"][0]
                                and "content" in response["choices"][0]["message"]
                            ):
                                result = response["choices"][0]["message"]["content"]
                                if self.debug:
                                    print(
                                        f"Response content from dict (first 100 chars): {result[:100]}..."
                                    )

                                if return_metadata:
                                    return result, metadata
                                return result

                    # Если ничего не нашли, но есть response в строковом виде
                    if isinstance(response, str):
                        if return_metadata:
                            return response, metadata
                        return response

                    # Последняя попытка получить ответ
                    if hasattr(response, "content"):
                        if return_metadata:
                            return response.content, metadata
                        return response.content

                    # Если все методы не сработали, возвращаем строку с ошибкой формата
                    error_msg = f"Failed to extract response content. Response type: {type(response)}"
                    if self.debug:
                        print(error_msg)
                        print(f"Response dump: {response}")

                    if return_metadata:
                        return error_msg, metadata
                    return error_msg

                except Exception as content_error:
                    if self.debug:
                        print(
                            f"Error extracting content from response: {str(content_error)}"
                        )
                    # Возвращаем сообщение об ошибке если не можем извлечь контент
                    error_msg = (
                        f"Error extracting response content: {str(content_error)}"
                    )

                    if return_metadata:
                        return error_msg, metadata
                    return error_msg

        except Exception as e:
            error_msg = (
                f"\nError during API call:"
                f"\nModel: {self.model_name}"
                f"\nAPI Type: {self.api_type}"
                f"\nBase URL: {self.base_url}"
            )
            if self.api_key:
                error_msg += f"\nAPI Key (first 8 chars): {self.api_key[:8]}..."
            error_msg += f"\nError: {str(e)}"

            print(error_msg)
            raise Exception(error_msg) from e
