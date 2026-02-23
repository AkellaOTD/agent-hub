"""Абстрактный интерфейс LLM провайдера."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    def __init__(self, model: str = ""):
        self.model = model

    @abstractmethod
    def chat(self, messages: list[dict], system: str, max_tokens: int = 4096) -> str:
        """Отправить сообщения и получить текстовый ответ."""

    @abstractmethod
    def provider_name(self) -> str:
        """Имя провайдера."""
