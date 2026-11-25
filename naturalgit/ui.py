"""
Terminal UI and color formatting for NaturalGit.
"""
import sys

# Force UTF-8 on Windows terminal for box-drawing characters and emojis
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Attempt to initialize colorama for full Windows terminal ANSI color support
try:
    import colorama
    colorama.init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright foreground
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Backgrounds
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


def header():
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}┌────────────────────────────────────────────────────────────┐{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}│{Colors.RESET}  🌿 {Colors.BOLD}{Colors.BRIGHT_WHITE}NaturalGit{Colors.RESET} - {Colors.DIM}Git con linguaggio naturale v1.0{Colors.RESET}          {Colors.BOLD}{Colors.BRIGHT_CYAN}│{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}└────────────────────────────────────────────────────────────┘{Colors.RESET}")


def print_status(branch, clean, staged_count, unstaged_count, untracked_count):
    status_parts = []
    if clean:
        status_parts.append(f"{Colors.GREEN}✔ Working tree pulito{Colors.RESET}")
    else:
        if staged_count > 0:
            status_parts.append(f"{Colors.GREEN}{staged_count} in staging{Colors.RESET}")
        if unstaged_count > 0:
            status_parts.append(f"{Colors.YELLOW}{unstaged_count} modificati{Colors.RESET}")
        if untracked_count > 0:
            status_parts.append(f"{Colors.RED}{untracked_count} non tracciati{Colors.RESET}")
    
    status_str = ", ".join(status_parts) if status_parts else "N/A"
    branch_display = f"{Colors.BRIGHT_MAGENTA}{branch}{Colors.RESET}" if branch else f"{Colors.DIM}Nessun repo{Colors.RESET}"
    print(f" {Colors.DIM}Branch:{Colors.RESET} {branch_display}  {Colors.DIM}|  Stato:{Colors.RESET} {status_str}\n")


def print_commit_preview(message: str):
    """Prints a styled box preview of a generated commit message."""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_MAGENTA}┌─ 📝 Anteprima Messaggio di Commit ────────────────────────┐{Colors.RESET}")
    for line in message.strip().splitlines():
        if line.startswith("Autore:") or line.startswith("Host/Sistema:") or line.startswith("Data:") or line.startswith("Branch:"):
            print(f"{Colors.BRIGHT_MAGENTA}│{Colors.RESET}  {Colors.CYAN}{line}{Colors.RESET}")
        elif line.startswith("File coinvolti") or line.startswith("Riepilogo:"):
            print(f"{Colors.BRIGHT_MAGENTA}│{Colors.RESET}  {Colors.BOLD}{Colors.YELLOW}{line}{Colors.RESET}")
        elif line.strip().startswith("-"):
            print(f"{Colors.BRIGHT_MAGENTA}│{Colors.RESET}    {Colors.WHITE}{line.strip()}{Colors.RESET}")
        else:
            print(f"{Colors.BRIGHT_MAGENTA}│{Colors.RESET}  {line}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_MAGENTA}└────────────────────────────────────────────────────────────┘{Colors.RESET}\n")


def print_proposal(command, explanation, engine_name="Regole locali", is_destructive=False, preview: str = ""):
    width = 62
    if preview:
        print_commit_preview(preview)
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}┌─ 🎯 Comando Identificato ({engine_name}) {'─' * max(0, width - 32 - len(engine_name))}┐{Colors.RESET}")
    
    # Command line
    cmd_lines = command.split(" && ")
    for i, cmd in enumerate(cmd_lines):
        prefix = "│  " if i == 0 else "│    && "
        print(f"{Colors.BRIGHT_BLUE}│{Colors.RESET}  {Colors.BOLD}{Colors.BRIGHT_GREEN}$ {cmd}{Colors.RESET}")
    
    print(f"{Colors.BRIGHT_BLUE}├────────────────────────────────────────────────────────────┤{Colors.RESET}")
    print(f"{Colors.BRIGHT_BLUE}│{Colors.RESET}  💡 {Colors.WHITE}{explanation}{Colors.RESET}")
    
    if is_destructive:
        print(f"{Colors.BRIGHT_BLUE}├────────────────────────────────────────────────────────────┤{Colors.RESET}")
        print(f"{Colors.BRIGHT_BLUE}│{Colors.RESET}  {Colors.BOLD}{Colors.BRIGHT_RED}⚠️  ATTENZIONE:{Colors.RESET} {Colors.RED}Operazione distruttiva o irreversibile!{Colors.RESET}")
        print(f"{Colors.BRIGHT_BLUE}│{Colors.RESET}     {Colors.DIM}Assicurati di non perdere modifiche non salvate.{Colors.RESET}")
    
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}└────────────────────────────────────────────────────────────┘{Colors.RESET}")


def print_success(msg):
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}✔ {msg}{Colors.RESET}")


def print_info(msg):
    print(f"{Colors.BRIGHT_CYAN}ℹ {msg}{Colors.RESET}")


def print_warning(msg):
    print(f"{Colors.BOLD}{Colors.BRIGHT_YELLOW}⚠ {msg}{Colors.RESET}")


def print_error(msg):
    print(f"{Colors.BOLD}{Colors.BRIGHT_RED}✖ {msg}{Colors.RESET}")
