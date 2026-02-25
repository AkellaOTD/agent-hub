"""Управление сессиями: директории, JSONL, Markdown."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class Message:
    id: str
    timestamp: str
    agent: str  # gpt | claude | gemini | human
    type: str   # conclusion | thinking | human | system
    content: str
    round: int = 0
    reply_to: str | None = None
    artifacts: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class Session:
    def __init__(self, session_dir: str, agents: list[str], artifacts_subdir: str = "artifacts"):
        self.dir = Path(session_dir)
        self.agents = agents
        self.artifacts_subdir = artifacts_subdir
        self._msg_counter = 0

    def create(self, task: str, config_path: str | None = None):
        """Создать новую сессию с начальной задачей."""
        self.dir.mkdir(parents=True, exist_ok=True)

        # Директории агентов
        for agent in self.agents:
            (self.dir / "agents" / agent).mkdir(parents=True, exist_ok=True)

        # Артефакты
        (self.dir / self.artifacts_subdir).mkdir(parents=True, exist_ok=True)

        # Копия конфига
        if config_path:
            shutil.copy2(config_path, self.dir / "config.yaml")

        # Задача
        (self.dir / "task.md").write_text(task, encoding='utf-8')

        # Инициализация shared
        (self.dir / "shared.jsonl").touch()
        (self.dir / "shared.md").write_text(f"# Сессия планирования\n\n## Задача\n\n{task}\n\n---\n\n", encoding='utf-8')

        # Первое сообщение — задача
        self.append_shared(Message(
            id=self._next_id("human"),
            timestamp=self._now(),
            agent="human",
            type="human",
            content=task,
            round=0,
        ))

    @classmethod
    def resume(cls, session_dir: str) -> 'Session':
        """Возобновить существующую сессию."""
        path = Path(session_dir)
        if not path.exists():
            raise FileNotFoundError(f"Сессия не найдена: {session_dir}")

        # Определяем агентов по директориям
        agents_dir = path / "agents"
        agents = [d.name for d in sorted(agents_dir.iterdir()) if d.is_dir()] if agents_dir.exists() else []

        session = cls(session_dir, agents)

        # Восстановим счётчик сообщений
        shared = session.read_shared()
        if shared:
            session._msg_counter = len(shared)

        return session

    def append_shared(self, msg: Message):
        """Добавить сообщение в общий чат (shared.jsonl + shared.md)."""
        # JSONL
        with open(self.dir / "shared.jsonl", 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(msg), ensure_ascii=False) + '\n')

        # Markdown
        with open(self.dir / "shared.md", 'a', encoding='utf-8') as f:
            label = msg.agent.upper() if msg.agent != "human" else "ADMIN"
            f.write(f"### [{label}] Раунд {msg.round}\n\n")
            f.write(msg.content + "\n\n---\n\n")

    def append_agent_context(self, agent_name: str, msg: Message):
        """Добавить запись в контекст агента."""
        ctx_file = self.dir / "agents" / agent_name / "context.jsonl"
        with open(ctx_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(msg), ensure_ascii=False) + '\n')

    def read_shared(self) -> list[Message]:
        """Прочитать все сообщения из shared.jsonl."""
        return self._read_jsonl(self.dir / "shared.jsonl")

    def read_agent_context(self, agent_name: str) -> list[Message]:
        """Прочитать контекст конкретного агента."""
        return self._read_jsonl(self.dir / "agents" / agent_name / "context.jsonl")

    def _safe_artifact_path(self, filename: str) -> Path:
        """Возвращает безопасный путь для артефакта, запрещая path traversal."""
        safe_name = Path(filename).name
        if not safe_name or safe_name in ('.', '..'):
            raise ValueError(f"Недопустимое имя артефакта: {filename!r}")
        if safe_name != filename:
            raise ValueError(f"Имя артефакта содержит путь: {filename!r} (допускаются только плоские имена)")
        artifact_path = (self.dir / self.artifacts_subdir / safe_name).resolve()
        artifacts_root = (self.dir / self.artifacts_subdir).resolve()
        if not str(artifact_path).startswith(str(artifacts_root) + "/") and artifact_path != artifacts_root:
            raise ValueError(f"Путь артефакта выходит за пределы: {filename!r}")
        return artifact_path

    def save_artifact(self, filename: str, content: str) -> Path:
        """Сохранить артефакт (план, схему и т.д.)."""
        artifact_path = self._safe_artifact_path(filename)
        artifact_path.write_text(content, encoding='utf-8')
        return artifact_path

    def read_artifact(self, filename: str) -> str | None:
        """Прочитать артефакт."""
        try:
            artifact_path = self._safe_artifact_path(filename)
        except ValueError:
            return None
        if artifact_path.exists():
            return artifact_path.read_text(encoding='utf-8')
        return None

    def list_artifacts(self) -> list[str]:
        """Список артефактов."""
        arts_dir = self.dir / self.artifacts_subdir
        if not arts_dir.exists():
            return []
        return [f.name for f in sorted(arts_dir.iterdir()) if f.is_file()]

    def get_task(self) -> str:
        """Прочитать исходную задачу."""
        task_file = self.dir / "task.md"
        if task_file.exists():
            return task_file.read_text(encoding='utf-8')
        return ""

    def save_dirs(self, project_dir: str, allowed_directories: list[str]):
        """Сохранить рабочие директории в сессию."""
        data = {"project_dir": project_dir, "allowed_directories": allowed_directories}
        (self.dir / "dirs.json").write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

    def load_dirs(self) -> tuple[str, list[str]]:
        """Загрузить рабочие директории из сессии."""
        path = self.dir / "dirs.json"
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            return data.get("project_dir", ""), data.get("allowed_directories", [])
        return "", []

    def _next_id(self, agent: str) -> str:
        self._msg_counter += 1
        return f"msg-{self._msg_counter:03d}-{agent}"

    def next_id(self, agent: str) -> str:
        return self._next_id(agent)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec='seconds')

    @staticmethod
    def now() -> str:
        return Session._now()

    @staticmethod
    def _read_jsonl(path: Path) -> list[Message]:
        msgs = []
        if not path.exists():
            return msgs
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    msgs.append(Message(**data))
        return msgs

    @staticmethod
    def generate_session_id(task: str) -> str:
        """Генерирует ID сессии из даты и первых слов задачи."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        # Берём первые 3-4 значимых слова
        words = task.lower().split()
        slug_words = []
        for w in words:
            cleaned = ''.join(c for c in w if c.isalnum() or c == '-')
            if cleaned and len(cleaned) > 2:
                slug_words.append(cleaned)
            if len(slug_words) >= 4:
                break
        slug = '-'.join(slug_words) if slug_words else 'session'
        return f"{date_str}-{slug}"
