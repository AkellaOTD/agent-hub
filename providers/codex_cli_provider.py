"""OpenAI Codex CLI провайдер."""

import re
import subprocess
from .base import LLMProvider
from .errors import ProviderError, QuotaExhaustedError, AuthError, TransientError


def _classify_codex_error(stderr: str) -> ProviderError:
    """Классифицирует stderr Codex CLI в типизированное исключение."""
    lower = stderr.lower()

    # Transient — проверяем первым
    transient_patterns = [
        r'timeout', r'timed?\s*out',
        r'connect', r'connection\s+(refused|reset|closed)',
        r'\b503\b', r'\b500\b', r'\b502\b', r'\b504\b',
        r'network\s+error', r'econnrefused', r'econnreset',
        r'server\s+error',
    ]
    if any(re.search(p, lower) for p in transient_patterns):
        return TransientError(f"codex CLI: временная ошибка", raw_stderr=stderr)

    # Auth
    auth_patterns = [r'\b401\b', r'\b403\b', r'api[_\s]?key', r'authentication', r'unauthorized', r'invalid.*key']
    if any(re.search(p, lower) for p in auth_patterns):
        return AuthError(f"codex CLI: ошибка авторизации", raw_stderr=stderr)

    # Quota
    quota_patterns = [r'\b429\b', r'quota', r'rate\s*limit', r'billing', r'insufficient.*quota']
    if any(re.search(p, lower) for p in quota_patterns):
        return QuotaExhaustedError(f"codex CLI: исчерпана квота", raw_stderr=stderr)

    # Всё остальное
    first_line = stderr.strip().split('\n')[0]
    return ProviderError(f"codex CLI: {first_line}", raw_stderr=stderr)


class CodexCLIProvider(LLMProvider):
    def chat(self, messages: list[dict], system: str, max_tokens: int = 4096) -> str:
        prompt = self._build_prompt(messages, system)

        # Промпт через stdin (передаём "-" как аргумент prompt)
        cmd = ["codex", "exec", "--skip-git-repo-check", "-"]
        if self.model:
            cmd.extend(["--model", self.model])

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=self.cwd,
            )
        except subprocess.TimeoutExpired:
            raise TransientError("codex CLI: таймаут выполнения (600с)", raw_stderr="")

        if result.returncode != 0:
            raise _classify_codex_error(result.stderr.strip())

        return result.stdout.strip()

    def _build_prompt(self, messages: list[dict], system: str) -> str:
        parts = [f"<system>\n{system}\n</system>\n"]
        for msg in messages:
            role = msg["role"].upper()
            parts.append(f"[{role}]\n{msg['content']}\n")
        return "\n".join(parts)

    def provider_name(self) -> str:
        return "codex_cli"
