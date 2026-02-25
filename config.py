"""Загрузка и валидация конфигурации."""

import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    name: str
    provider: str  # claude_cli | gemini_cli | codex_cli
    model: str
    system_prompt: str
    max_tokens: int = 4096


@dataclass
class Config:
    agents: dict[str, AgentConfig] = field(default_factory=dict)
    turn_order: list[str] = field(default_factory=list)
    max_rounds: int = 10
    review_peers: bool = True
    sessions_dir: str = "./sessions"
    artifacts_dir: str = "artifacts"
    max_shared_messages: int = 0       # 0 = без лимита
    max_own_thinking_chars: int = 0    # 0 = без лимита
    max_peer_thinking_chars: int = 0   # 0 = без лимита
    project_dir: str = ""              # рабочая директория проекта (cwd для агентов)


VALID_PROVIDERS = ('claude_cli', 'gemini_cli', 'codex_cli')


def load_config(path: str) -> Config:
    """Загружает конфиг из YAML файла."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    agents = {}
    for name, agent_raw in raw.get('agents', {}).items():
        agents[name] = AgentConfig(
            name=name,
            provider=agent_raw['provider'],
            model=agent_raw.get('model', ''),
            system_prompt=agent_raw.get('system_prompt', ''),
            max_tokens=agent_raw.get('max_tokens', 4096),
        )

    turn_order = raw.get('turn_order', list(agents.keys()))

    for agent_name in turn_order:
        if agent_name not in agents:
            raise ValueError(f"Агент '{agent_name}' в turn_order, но не определён в agents")

    for agent in agents.values():
        if agent.provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Неизвестный провайдер '{agent.provider}' для агента '{agent.name}'. "
                f"Доступные: {', '.join(VALID_PROVIDERS)}"
            )

    return Config(
        agents=agents,
        turn_order=turn_order,
        max_rounds=raw.get('max_rounds', 10),
        review_peers=raw.get('review_peers', True),
        sessions_dir=raw.get('sessions_dir', './sessions'),
        artifacts_dir=raw.get('artifacts_dir', 'artifacts'),
        max_shared_messages=raw.get('max_shared_messages', 0),
        max_own_thinking_chars=raw.get('max_own_thinking_chars', 0),
        max_peer_thinking_chars=raw.get('max_peer_thinking_chars', 0),
        project_dir=raw.get('project_dir', ''),
    )
