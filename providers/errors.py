"""Типизированные исключения провайдеров."""


class ProviderError(Exception):
    """Базовая ошибка провайдера с сохранением сырого stderr."""

    def __init__(self, message: str, raw_stderr: str = ""):
        super().__init__(message)
        self.raw_stderr = raw_stderr


class QuotaExhaustedError(ProviderError):
    """Исчерпана квота/лимит API."""


class AuthError(ProviderError):
    """Ошибка авторизации (невалидный ключ, 401/403)."""


class TransientError(ProviderError):
    """Временная ошибка (таймаут, сетевой сбой, 5xx)."""
