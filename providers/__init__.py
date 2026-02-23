from .base import LLMProvider
from .errors import ProviderError, QuotaExhaustedError, AuthError, TransientError
from .claude_cli_provider import ClaudeCLIProvider
from .gemini_cli_provider import GeminiCLIProvider
from .codex_cli_provider import CodexCLIProvider

PROVIDERS = {
    'claude_cli': ClaudeCLIProvider,
    'gemini_cli': GeminiCLIProvider,
    'codex_cli': CodexCLIProvider,
}


def create_provider(provider_name: str, model: str) -> LLMProvider:
    """Фабрика провайдеров."""
    if provider_name not in PROVIDERS:
        available = ', '.join(sorted(PROVIDERS.keys()))
        raise ValueError(f"Неизвестный провайдер: {provider_name}. Доступные: {available}")
    return PROVIDERS[provider_name](model=model)
