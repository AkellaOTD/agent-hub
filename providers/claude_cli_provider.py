"""Claude Code CLI провайдер."""

import re
import subprocess
import os
from .base import LLMProvider
from .errors import ProviderError, QuotaExhaustedError, AuthError, TransientError


def _classify_claude_error(stderr: str) -> ProviderError:
    """Классифицирует stderr Claude CLI в типизированное исключение."""
    lower = stderr.lower()

    # Transient — проверяем первым
    transient_patterns = [
        r'timeout', r'timed?\s*out',
        r'connect', r'connection\s+(refused|reset|closed)',
        r'\b503\b', r'\b500\b', r'\b502\b', r'\b504\b',
        r'network\s+error', r'econnrefused', r'econnreset',
        r'overloaded', r'server\s+error',
    ]
    if any(re.search(p, lower) for p in transient_patterns):
        return TransientError(f"claude CLI: временная ошибка", raw_stderr=stderr)

    # Auth
    auth_patterns = [r'\b401\b', r'\b403\b', r'api[_\s]?key', r'authentication', r'unauthorized', r'invalid.*key']
    if any(re.search(p, lower) for p in auth_patterns):
        return AuthError(f"claude CLI: ошибка авторизации", raw_stderr=stderr)

    # Quota
    quota_patterns = [r'\b429\b', r'quota', r'rate\s*limit', r'credit', r'billing', r'usage\s*limit']
    if any(re.search(p, lower) for p in quota_patterns):
        return QuotaExhaustedError(f"claude CLI: исчерпана квота", raw_stderr=stderr)

    # Всё остальное
    first_line = stderr.strip().split('\n')[0]
    return ProviderError(f"claude CLI: {first_line}", raw_stderr=stderr)


class ClaudeCLIProvider(LLMProvider):
    def chat(self, messages: list[dict], system: str, max_tokens: int = 4096) -> str:
        prompt = self._build_prompt(messages, system)

        env = os.environ.copy()
        env.pop("CLAUDECODE", None)

        # Промпт через stdin
        cmd = ["claude", "-p"]
        if self.model:
            cmd.extend(["--model", self.model])

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
                cwd=self.cwd,
            )
        except subprocess.TimeoutExpired:
            raise TransientError("claude CLI: таймаут выполнения (600с)", raw_stderr="")

        if result.returncode != 0:
            raise _classify_claude_error(result.stderr.strip())

        return result.stdout.strip()

    def _build_prompt(self, messages: list[dict], system: str) -> str:
        parts = [f"<system>\n{system}\n</system>\n"]
        for msg in messages:
            role = msg["role"].upper()
            parts.append(f"[{role}]\n{msg['content']}\n")
        return "\n".join(parts)

    def provider_name(self) -> str:
        return "claude_cli"
