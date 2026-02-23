#!/usr/bin/env python3
"""Multi-Agent Chat Orchestrator — главный модуль."""

import argparse
import re
import sys
import time
from pathlib import Path

from config import load_config, Config
from session import Session, Message
from context import build_system_prompt, build_messages
from providers import create_provider
from providers.errors import ProviderError, QuotaExhaustedError, AuthError, TransientError
from renderer import (
    Spinner,
    print_header, print_round_header,
    print_agent_conclusion, print_agent_thinking_summary,
    print_agent_question, print_question_prompt,
    print_artifact_saved, print_human_prompt, print_help,
    print_status, print_error, print_info, print_finalizing,
)

MAX_RETRIES = 2
RETRY_DELAY = 10  # секунд


def parse_response(response: str) -> tuple[str, str, list[tuple[str, str]], list[str]]:
    """Парсит ответ агента на thinking, conclusion, артефакты и вопросы.

    Returns:
        (thinking, conclusion, [(filename, content), ...], [questions])
    """
    thinking = ""
    conclusion = ""

    # Извлекаем [THINKING] блок
    thinking_match = re.search(
        r'\[THINKING\]\s*\n(.*?)(?=\[CONCLUSION\]|\Z)',
        response, re.DOTALL
    )
    if thinking_match:
        thinking = thinking_match.group(1).strip()

    # Извлекаем [CONCLUSION] блок
    conclusion_match = re.search(
        r'\[CONCLUSION\]\s*\n(.*)',
        response, re.DOTALL
    )
    if conclusion_match:
        conclusion = conclusion_match.group(1).strip()

    # Если формат не соблюдён — весь ответ в conclusion
    if not thinking and not conclusion:
        conclusion = response.strip()

    # Извлекаем артефакты из conclusion
    artifacts = []
    artifact_pattern = re.compile(
        r'\[ARTIFACT:([^\]]+)\]\s*\n(.*?)\[/ARTIFACT\]',
        re.DOTALL
    )
    for match in artifact_pattern.finditer(conclusion):
        filename = match.group(1).strip()
        content = match.group(2).strip()
        artifacts.append((filename, content))

    # Извлекаем вопросы из conclusion
    questions = []
    question_pattern = re.compile(
        r'\[QUESTION\]\s*\n(.*?)\[/QUESTION\]',
        re.DOTALL
    )
    for match in question_pattern.finditer(conclusion):
        questions.append(match.group(1).strip())

    # Убираем артефакты и вопросы из текста conclusion
    clean_conclusion = artifact_pattern.sub('', conclusion)
    clean_conclusion = question_pattern.sub('', clean_conclusion).strip()

    # Эвристика: если тегов [QUESTION] нет, ищем вопросы в тексте
    if not questions:
        questions = _detect_natural_questions(clean_conclusion)

    return thinking, clean_conclusion, artifacts, questions


def _detect_natural_questions(text: str) -> list[str]:
    """Обнаруживает вопросы к администратору без явных тегов."""
    # Ищем блоки типа "Вопросы к администратору:", "Вопросы:", "Questions:" и т.п.
    block_pattern = re.compile(
        r'(?:вопрос[ыа]?\s*(?:к\s+администратору|к\s+админу|для\s+финализации|для\s+уточнения)?[:\s]*\n)'
        r'((?:\s*\d+[\.\)]\s*.+\n?)+)',
        re.IGNORECASE | re.MULTILINE
    )
    match = block_pattern.search(text)
    if match:
        block = match.group(1).strip()
        # Извлекаем нумерованные вопросы
        items = re.findall(r'\d+[\.\)]\s*(.+)', block)
        if items:
            return ['\n'.join(items)]

    return []


def _save_error_artifact(session: Session, agent_name: str, round_num: int, error: ProviderError):
    """Сохраняет сырой stderr ошибки провайдера в артефакт сессии."""
    if error.raw_stderr:
        filename = f"error-{agent_name}-round{round_num}.log"
        session.save_artifact(filename, error.raw_stderr)


def run_agent_turn(
    session: Session,
    agent_name: str,
    config: Config,
    round_num: int,
    no_interactive: bool = False,
) -> bool:
    """Выполняет ход одного агента. Возвращает True при успехе."""
    agent_config = config.agents[agent_name]

    spinner = Spinner(agent_name)
    spinner.start()
    start_time = time.time()

    try:
        # Создаём провайдер
        provider = create_provider(
            agent_config.provider,
            agent_config.model,
        )

        # Собираем контекст
        system_prompt = build_system_prompt(agent_config, round_num, config.turn_order)
        messages = build_messages(session, agent_name, round_num, config.review_peers)

        # Вызываем LLM с retry при transient-ошибках
        response = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = provider.chat(messages, system_prompt, agent_config.max_tokens)
                break
            except TransientError as e:
                if attempt < MAX_RETRIES:
                    spinner.stop()
                    print_info(f"{agent_name}: временная ошибка, повтор {attempt}/{MAX_RETRIES} через {RETRY_DELAY}с...")
                    _save_error_artifact(session, agent_name, round_num, e)
                    time.sleep(RETRY_DELAY)
                    spinner.start()
                else:
                    raise

        elapsed = time.time() - start_time
        spinner.stop()

        # Парсим ответ
        thinking, conclusion, artifacts, questions = parse_response(response)

        # Сохраняем thinking в контекст агента
        if thinking:
            session.append_agent_context(agent_name, Message(
                id=session.next_id(agent_name),
                timestamp=Session.now(),
                agent=agent_name,
                type="thinking",
                content=thinking,
                round=round_num,
            ))

        # Сохраняем conclusion в shared
        artifact_names = [a[0] for a in artifacts]
        session.append_shared(Message(
            id=session.next_id(agent_name),
            timestamp=Session.now(),
            agent=agent_name,
            type="conclusion",
            content=conclusion,
            round=round_num,
            artifacts=artifact_names,
        ))

        # Сохраняем артефакты
        for filename, content in artifacts:
            session.save_artifact(filename, content)
            print_artifact_saved(filename)

        # Выводим в CLI
        print_agent_conclusion(agent_name, conclusion, elapsed)
        if thinking:
            print_agent_thinking_summary(agent_name, thinking, elapsed)

        # Если агент задал вопросы — спросить админа сразу
        if questions and not no_interactive:
            print_agent_question(agent_name, questions)
            print_question_prompt()
            try:
                answer = input().strip()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer:
                session.append_shared(Message(
                    id=session.next_id("human"),
                    timestamp=Session.now(),
                    agent="human",
                    type="human",
                    content=f"Ответ на вопрос {agent_name.upper()}: {answer}",
                    round=round_num,
                ))
                print_info("Ответ добавлен в общий чат")

        return True

    except QuotaExhaustedError as e:
        spinner.stop()
        _save_error_artifact(session, agent_name, round_num, e)
        print_error(f"{agent_name}: исчерпана квота API. Подождите или переключитесь на платный тариф.")
        return False

    except AuthError as e:
        spinner.stop()
        _save_error_artifact(session, agent_name, round_num, e)
        print_error(f"{agent_name}: ошибка авторизации. Проверьте API ключ.")
        return False

    except TransientError as e:
        spinner.stop()
        _save_error_artifact(session, agent_name, round_num, e)
        print_error(f"{agent_name}: ошибка соединения после {MAX_RETRIES} попыток.")
        return False

    except ProviderError as e:
        spinner.stop()
        _save_error_artifact(session, agent_name, round_num, e)
        print_error(f"{agent_name}: {e}")
        return False

    except Exception as e:
        spinner.stop()
        print_error(f"{agent_name}: {str(e).split(chr(10))[0]}")
        return False


def handle_human_input(session: Session, round_num: int, config: Config) -> str:
    """Обрабатывает ввод человека. Возвращает действие: 'continue', 'done', 'quit'."""
    while True:
        print_human_prompt()
        try:
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 'quit'

        if not user_input:
            return 'continue'

        if user_input == '/done':
            return 'done'

        if user_input == '/quit' or user_input == '/exit':
            return 'quit'

        if user_input == '/help':
            print_help()
            continue

        if user_input == '/status':
            shared = session.read_shared()
            artifacts = session.list_artifacts()
            print_status(round_num, config.max_rounds, config.turn_order, len(shared), artifacts)
            continue

        if user_input.startswith('/read '):
            agent = user_input[6:].strip().lower()
            if agent in config.turn_order:
                ctx = session.read_agent_context(agent)
                if ctx:
                    for msg in ctx:
                        print(f"\n--- Раунд {msg.round} ---")
                        print(msg.content)
                else:
                    print_info(f"Контекст {agent.upper()} пуст")
            else:
                print_error(f"Агент '{agent}' не найден. Доступные: {', '.join(config.turn_order)}")
            continue

        if user_input == '/artifacts':
            artifacts = session.list_artifacts()
            if artifacts:
                for a in artifacts:
                    print(f"  - {a}")
            else:
                print_info("Артефактов пока нет")
            continue

        if user_input.startswith('/artifact '):
            name = user_input[10:].strip()
            content = session.read_artifact(name)
            if content:
                print(f"\n--- {name} ---")
                print(content)
            else:
                print_error(f"Артефакт '{name}' не найден")
            continue

        # Обычный текст — комментарий в чат
        session.append_shared(Message(
            id=session.next_id("human"),
            timestamp=Session.now(),
            agent="human",
            type="human",
            content=user_input,
            round=round_num,
        ))
        print_info("Комментарий добавлен в общий чат")
        return 'continue'


def finalize_session(session: Session, config: Config):
    """Формирует итоговый план из всех выводов."""
    print_finalizing()

    shared = session.read_shared()
    conclusions = [m for m in shared if m.type == "conclusion"]

    if not conclusions:
        print_info("Нет выводов для финализации")
        return

    # Собираем итоговый план
    parts = ["# Итоговый план\n"]
    parts.append(f"## Задача\n\n{session.get_task()}\n")

    # Группируем по раундам
    rounds: dict[int, list[Message]] = {}
    for msg in conclusions:
        rounds.setdefault(msg.round, []).append(msg)

    for round_num in sorted(rounds.keys()):
        parts.append(f"\n## Раунд {round_num}\n")
        for msg in rounds[round_num]:
            parts.append(f"### {msg.agent.upper()}\n\n{msg.content}\n")

    # Артефакты
    artifacts = session.list_artifacts()
    if artifacts:
        parts.append("\n## Артефакты\n")
        for name in artifacts:
            content = session.read_artifact(name)
            parts.append(f"### {name}\n\n{content}\n")

    plan_content = "\n".join(parts)
    path = session.save_artifact("plan-final.md", plan_content)
    print_info(f"Итоговый план сохранён: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Chat Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python orchestrator.py --config config.yaml --task "Спланировать систему авторизации"
  python orchestrator.py --config config.yaml
  python orchestrator.py --resume sessions/2026-02-23-auth/
        """
    )
    parser.add_argument('--config', '-c', default='config.yaml', help='Путь к конфигу (default: config.yaml)')
    parser.add_argument('--task', '-t', help='Задача (или ввести интерактивно)')
    parser.add_argument('--resume', '-r', help='Путь к существующей сессии для продолжения')
    parser.add_argument('--rounds', type=int, help='Количество раундов (переопределяет конфиг)')
    parser.add_argument('--no-interactive', action='store_true', help='Без интерактивного ввода')

    args = parser.parse_args()

    # Загрузка конфига
    config_path = args.config
    if args.resume:
        resume_config = Path(args.resume) / "config.yaml"
        if resume_config.exists():
            config_path = str(resume_config)

    try:
        config = load_config(config_path)
    except Exception as e:
        print_error(f"Ошибка конфига: {e}")
        sys.exit(1)

    if args.rounds:
        config.max_rounds = args.rounds

    # Сессия
    if args.resume:
        session = Session.resume(args.resume)
        task = session.get_task()
        print_header(f"Продолжение сессии: {args.resume}")
        print_info(f"Задача: {task[:100]}...")
        # Определяем текущий раунд
        shared = session.read_shared()
        conclusions = [m for m in shared if m.type == "conclusion"]
        start_round = (len(conclusions) // len(config.turn_order)) + 1
    else:
        # Получаем задачу
        task = args.task
        if not task:
            print_header("Multi-Agent Chat Orchestrator")
            print("Введите задачу для обсуждения (Ctrl+D для завершения ввода):\n")
            lines = []
            try:
                while True:
                    lines.append(input())
            except EOFError:
                pass
            task = "\n".join(lines).strip()

        if not task:
            print_error("Задача не указана")
            sys.exit(1)

        session_id = Session.generate_session_id(task)
        session_dir = str(Path(config.sessions_dir) / session_id)

        session = Session(session_dir, config.turn_order, config.artifacts_dir)
        session.create(task, args.config)

        print_header("Multi-Agent Chat Orchestrator")
        print_info(f"Сессия: {session_dir}")
        print_info(f"Задача: {task[:100]}...")
        print_info(f"Агенты: {', '.join(a.upper() for a in config.turn_order)}")
        print_info(f"Раундов: {config.max_rounds}")
        start_round = 1

    # Основной цикл
    for round_num in range(start_round, config.max_rounds + 1):
        print_round_header(round_num)

        # Ход каждого агента
        for agent_name in config.turn_order:
            success = run_agent_turn(session, agent_name, config, round_num, args.no_interactive)
            if not success:
                print_error(f"Ход {agent_name.upper()} провалился, пропускаю")

        # Ввод человека (если не --no-interactive)
        if not args.no_interactive:
            action = handle_human_input(session, round_num, config)
            if action == 'done':
                finalize_session(session, config)
                break
            elif action == 'quit':
                print_info("Сессия приостановлена. Используйте --resume для продолжения.")
                break
        # В no-interactive режиме продолжаем автоматически

    else:
        # Достигнут лимит раундов
        print_info(f"\nДостигнут лимит в {config.max_rounds} раундов.")
        finalize_session(session, config)

    print_info(f"\nФайлы сессии: {session.dir}")


if __name__ == '__main__':
    main()
