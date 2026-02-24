"""CLI вывод с цветами и спиннером (ANSI, без внешних зависимостей)."""

import sys
import threading
import time

# ANSI цвета для агентов
AGENT_COLORS = {
    'gpt':     '\033[92m',   # зелёный
    'claude':  '\033[94m',   # синий
    'gemini':  '\033[93m',   # жёлтый
    'human':   '\033[97m',   # белый
    'system':  '\033[90m',   # серый
}

BOLD = '\033[1m'
DIM = '\033[2m'
RESET = '\033[0m'
CYAN = '\033[96m'
RED = '\033[91m'
MAGENTA = '\033[95m'
CLEAR_LINE = '\033[2K\r'

SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

AGENT_PHASES = [
    "принял задачу, анализирую контекст",
    "изучаю предложения коллег",
    "формирую рассуждения",
    "готовлю выводы",
]


def agent_color(agent_name: str) -> str:
    return AGENT_COLORS.get(agent_name, '\033[37m')


class Spinner:
    """Анимированный спиннер с фазами работы агента."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.color = agent_color(agent_name)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        sys.stdout.write(CLEAR_LINE)
        sys.stdout.flush()

    def _spin(self):
        start = time.time()
        frame_idx = 0
        while not self._stop.is_set():
            elapsed = int(time.time() - start)
            # Меняем фазу каждые 8 секунд
            phase_idx = min(elapsed // 8, len(AGENT_PHASES) - 1)
            phase = AGENT_PHASES[phase_idx]
            spinner = SPINNER_FRAMES[frame_idx % len(SPINNER_FRAMES)]

            line = (
                f"{CLEAR_LINE}{self.color}{BOLD}[{self.agent_name.upper()}]{RESET} "
                f"{self.color}{spinner} {phase}... {DIM}({elapsed}с){RESET}"
            )
            sys.stdout.write(line)
            sys.stdout.flush()

            frame_idx += 1
            self._stop.wait(0.1)


def print_header(text: str):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")


def print_round_header(round_num: int):
    print(f"\n{BOLD}{MAGENTA}--- Раунд {round_num} ---{RESET}\n")


def print_agent_conclusion(agent_name: str, conclusion: str, elapsed: float):
    color = agent_color(agent_name)
    print(f"{color}{BOLD}[{agent_name.upper()}]{RESET} {DIM}ответил за {elapsed:.0f}с{RESET}")
    for line in conclusion.split('\n'):
        print(f"  {color}{line}{RESET}")
    print()


def print_agent_thinking_summary(agent_name: str, thinking: str, elapsed: float):
    """Краткая информация о thinking (не полный текст)."""
    color = agent_color(agent_name)
    lines = thinking.strip().split('\n')
    word_count = len(thinking.split())
    print(f"  {DIM}{color}[thinking: {word_count} слов, {len(lines)} строк | думал на протяжении {elapsed:.0f} секунд]{RESET}")


def print_artifact_saved(filename: str):
    print(f"  {DIM}[artifact сохранён: {filename}]{RESET}")


def print_agent_question(agent_name: str, questions: list[str]):
    """Выводит вопросы агента к администратору."""
    color = agent_color(agent_name)
    print(f"\n  {color}{BOLD}[{agent_name.upper()} спрашивает]{RESET}")
    for q in questions:
        for line in q.split('\n'):
            print(f"  {color}  ? {line}{RESET}")
    print()


def print_question_prompt():
    """Приглашение для ответа на вопрос."""
    print(f"{BOLD}{CYAN}[ADMIN]{RESET} Ответьте на вопрос (Enter=пропустить):")
    print(f"{CYAN}> {RESET}", end='', flush=True)


def print_human_prompt():
    """Приглашение для ввода человека."""
    print(f"{BOLD}{CYAN}[ADMIN]{RESET} Ваш ход (Enter=продолжить, /done=завершить, /help=команды):")
    print(f"{CYAN}> {RESET}", end='', flush=True)


def print_help():
    print(f"""
{BOLD}Команды:{RESET}
  {CYAN}Enter{RESET}          — продолжить следующий раунд
  {CYAN}/done{RESET}          — завершить сессию, сформировать итоговый план
  {CYAN}/status{RESET}        — показать текущее состояние сессии
  {CYAN}/read <agent>{RESET}  — показать рассуждения агента (gpt, claude, gemini)
  {CYAN}/artifacts{RESET}     — список артефактов
  {CYAN}/artifact <name>{RESET} — показать содержимое артефакта
  {CYAN}/summary{RESET}        — краткая сводка выводов по раундам
  {CYAN}/help{RESET}          — эта справка
  {CYAN}текст{RESET}          — добавить комментарий/направление в общий чат
""")


def print_status(round_num: int, max_rounds: int, agents: list[str], msg_count: int, artifacts: list[str]):
    print(f"""
{BOLD}Статус сессии:{RESET}
  Раунд:      {round_num}/{max_rounds}
  Агенты:     {', '.join(a.upper() for a in agents)}
  Сообщений:  {msg_count}
  Артефактов: {len(artifacts)} ({', '.join(artifacts) if artifacts else 'нет'})
""")


def print_error(text: str):
    print(f"{RED}{BOLD}[ошибка]{RESET} {RED}{text}{RESET}")


def print_info(text: str):
    print(f"{DIM}{text}{RESET}")


def print_finalizing():
    print(f"\n{BOLD}{CYAN}Формирую итоговый план...{RESET}\n")
