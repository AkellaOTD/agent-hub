"""Gemini CLI провайдер."""

import re
import subprocess
from .base import LLMProvider
from .errors import ProviderError, QuotaExhaustedError, AuthError, TransientError


def _classify_gemini_error(stderr: str) -> ProviderError:
    """Классифицирует stderr Gemini CLI в типизированное исключение."""
    lower = stderr.lower()

    # Transient — проверяем ДО quota, чтобы "quota service unreachable" не попал в QuotaExhausted
    transient_patterns = [
        r'timeout', r'timed?\s*out', r'deadline_exceeded',
        r'connect', r'connection\s+(refused|reset|closed)',
        r'unavailable', r'\b503\b', r'\b500\b', r'\b502\b', r'\b504\b',
        r'network\s+error', r'econnrefused', r'econnreset',
    ]
    if any(re.search(p, lower) for p in transient_patterns):
        return TransientError(f"gemini CLI: временная ошибка", raw_stderr=stderr)

    # Auth
    auth_patterns = [r'\b401\b', r'\b403\b', r'api[_\s]?key', r'authentication', r'unauthorized', r'permission\s+denied']
    if any(re.search(p, lower) for p in auth_patterns):
        return AuthError(f"gemini CLI: ошибка авторизации", raw_stderr=stderr)

    # Quota — только после исключения transient
    quota_patterns = [r'resource_exhausted', r'\b429\b', r'quota', r'rate\s*limit']
    if any(re.search(p, lower) for p in quota_patterns):
        return QuotaExhaustedError(f"gemini CLI: исчерпана квота", raw_stderr=stderr)

    # Всё остальное
    first_line = stderr.strip().split('\n')[0]
    return ProviderError(f"gemini CLI: {first_line}", raw_stderr=stderr)


class GeminiCLIProvider(LLMProvider):
    def chat(self, messages: list[dict], system: str, max_tokens: int = 4096) -> str:
        prompt = self._build_prompt(messages, system)

        # Промпт через stdin, -p с минимальной инструкцией
        cmd = ["gemini", "-p", "Ответь на запрос из stdin"]
        if self.model:
            cmd.extend(["--model", self.model])

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            raise TransientError("gemini CLI: таймаут выполнения (600с)", raw_stderr="")

        if result.returncode != 0:
            raise _classify_gemini_error(result.stderr.strip())

        return result.stdout.strip()

    def _build_prompt(self, messages: list[dict], system: str) -> str:
        parts = [f"<system>\n{system}\n</system>\n"]
        for msg in messages:
            role = msg["role"].upper()
            parts.append(f"[{role}]\n{msg['content']}\n")
        return "\n".join(parts)

    def provider_name(self) -> str:
        return "gemini_cli"
