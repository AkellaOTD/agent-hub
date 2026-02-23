"""Сборка контекста для каждого агента перед вызовом LLM."""

from session import Session, Message
from config import AgentConfig

# Инструкция формата ответа, добавляется к system prompt каждого агента
RESPONSE_FORMAT_INSTRUCTION = """

=== ФОРМАТ ОТВЕТА ===
Твой ответ ДОЛЖЕН содержать строго два блока:

[THINKING]
Здесь твои полные рассуждения, анализ, сомнения, альтернативы.
Это записывается в твой личный контекст. Другие агенты МОГУТ это прочитать
и указать на ошибки в твоих рассуждениях.

[CONCLUSION]
Здесь только конкретные выводы и предложения для общего обсуждения.
Это видят все участники. Будь лаконичен и конструктивен.

Если создаёшь артефакт (план, схему), оберни его в блок внутри [CONCLUSION]:
[ARTIFACT:filename.md]
содержимое артефакта
[/ARTIFACT]

Если тебе нужна информация от администратора (человека) для продолжения работы —
задай вопрос в блоке внутри [CONCLUSION]:
[QUESTION]
Твой вопрос к администратору. Можно несколько вопросов, каждый с новой строки.
[/QUESTION]
Администратор увидит вопрос и ответит до того, как начнёт работать следующий агент.

Оба блока обязательны. Начинай ответ с [THINKING].
"""


def build_system_prompt(agent_config: AgentConfig, round_num: int, total_agents: list[str]) -> str:
    """Собирает полный system prompt для агента."""
    peers = [a for a in total_agents if a != agent_config.name]
    peer_list = ", ".join(peers)

    header = (
        f"Ты — {agent_config.name.upper()}, участник команды экспертного планирования.\n"
        f"Другие участники: {peer_list}.\n"
        f"Текущий раунд: {round_num}.\n"
    )

    if round_num == 1:
        header += "Это первый раунд — предложи свой вариант плана.\n"
    else:
        header += (
            "Это не первый раунд. Учитывай предложения других агентов, "
            "ищи пробелы, противоречия, предлагай улучшения. "
            "Двигайся к консенсусу.\n"
        )

    return header + "\n" + agent_config.system_prompt + RESPONSE_FORMAT_INSTRUCTION


def build_messages(
    session: Session,
    agent_name: str,
    round_num: int,
    review_peers: bool = True,
) -> list[dict]:
    """Собирает список сообщений для LLM API.

    Контекст состоит из:
    1. Задача (первое сообщение)
    2. Общий чат (выводы всех участников)
    3. Свой контекст рассуждений
    4. Контексты пиров (если review_peers=True)
    """
    messages = []

    # 1. Задача
    task = session.get_task()
    messages.append({
        "role": "user",
        "content": f"## Задача\n\n{task}"
    })

    # 2. Общий чат
    shared = session.read_shared()
    if len(shared) > 1:  # Первое сообщение — задача, уже добавлена
        shared_text = _format_shared(shared[1:])  # Пропускаем задачу
        messages.append({
            "role": "user",
            "content": f"## Общий чат (выводы участников)\n\n{shared_text}"
        })

    # 3. Свой контекст
    own_context = session.read_agent_context(agent_name)
    if own_context:
        own_text = _format_agent_context(own_context)
        messages.append({
            "role": "user",
            "content": f"## Твои предыдущие рассуждения\n\n{own_text}"
        })

    # 4. Контексты пиров
    if review_peers:
        for peer in session.agents:
            if peer == agent_name:
                continue
            peer_context = session.read_agent_context(peer)
            if peer_context:
                # Берём только последнюю запись пира для экономии токенов
                last = peer_context[-1]
                messages.append({
                    "role": "user",
                    "content": (
                        f"## Рассуждения {peer.upper()} (последний раунд)\n\n"
                        f"{last.content}"
                    )
                })

    # Финальный промпт
    messages.append({
        "role": "user",
        "content": f"Сейчас раунд {round_num}. Дай свой ответ в формате [THINKING] и [CONCLUSION]."
    })

    return messages


def _format_shared(messages: list[Message]) -> str:
    """Форматирует сообщения общего чата в текст."""
    parts = []
    for msg in messages:
        label = msg.agent.upper() if msg.agent != "human" else "ADMIN"
        parts.append(f"**[{label}, раунд {msg.round}]:**\n{msg.content}\n")
    return "\n---\n".join(parts)


def _format_agent_context(messages: list[Message]) -> str:
    """Форматирует контекст агента в текст."""
    parts = []
    for msg in messages:
        parts.append(f"**[Раунд {msg.round}]:**\n{msg.content}\n")
    return "\n---\n".join(parts)
