"""Абстрактный интерфейс LLM провайдера."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    def __init__(self, model: str = "", cwd: str = "", allowed_directories: list[str] | None = None):
        self.model = model
        self.cwd = cwd or None
        self.allowed_directories = allowed_directories or []

    @abstractmethod
    def chat(self, messages: list[dict], system: str, max_tokens: int = 4096) -> str:
        """Отправить сообщения и получить текстовый ответ."""

    @abstractmethod
    def provider_name(self) -> str:
        """Имя провайдера."""
