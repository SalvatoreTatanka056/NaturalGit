"""
Command executor with safety checks, confirmation, and interactive editing.
"""
import shlex
import subprocess
import sys
from typing import Optional
from .ui import Colors, print_proposal, print_success, print_error, print_warning, print_info


def run_command_live(command_str: str) -> int:
    """Executes a command or chained commands live, streaming output."""
    sub_commands = [c.strip() for c in command_str.split(" && ") if c.strip()]
    
    for cmd in sub_commands:
        print(f"\n{Colors.DIM}▶ Esecuzione:{Colors.RESET} {Colors.BOLD}{cmd}{Colors.RESET}")
        try:
            # We run via subprocess with shell=True on Windows to support chained commands and pipes
            ret = subprocess.call(cmd, shell=True)
            if ret != 0:
                print_error(f"Comando terminato con codice di errore: {ret}")
                return ret
        except Exception as e:
            print_error(f"Errore durante l'esecuzione: {e}")
            return -1

    return 0


def execute_proposal(
    command: str,
    explanation: str,
    engine_name: str = "Regole locali",
    is_destructive: bool = False,
    auto_confirm: bool = False,
    dry_run: bool = False,
    preview: str = ""
) -> bool:
    """
    Displays the proposed command and asks the user for confirmation.
    Returns True if executed successfully, False otherwise.
    """
    print_proposal(command, explanation, engine_name=engine_name, is_destructive=is_destructive, preview=preview)

    if dry_run:
        print_info("Modalità Dry-Run: nessun comando eseguito.")
        return True

    if auto_confirm and not is_destructive:
        print_info("Esecuzione automatica (--yes)...")
        code = run_command_live(command)
        if code == 0:
            print_success("Operazione completata con successo!")
            return True
        return False

    # Interactive confirmation prompt
    prompt_msg = (
        f"{Colors.BOLD}Eseguire questo comando? {Colors.RESET}"
        f"[{Colors.GREEN}S{Colors.RESET}i/{Colors.RED}n{Colors.RESET}o/"
        f"{Colors.YELLOW}m{Colors.RESET}odifica]: "
    )

    try:
        choice = input(prompt_msg).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\n" + Colors.DIM + "Operazione annullata." + Colors.RESET)
        return False

    if choice in ["", "s", "si", "y", "yes"]:
        code = run_command_live(command)
        if code == 0:
            print_success("Operazione completata con successo!")
            return True
        return False
    elif choice in ["m", "modifica", "edit"]:
        try:
            new_cmd = input(f"{Colors.BOLD}Modifica comando:{Colors.RESET} ").strip()
            if new_cmd:
                code = run_command_live(new_cmd)
                if code == 0:
                    print_success("Operazione completata!")
                    return True
                return False
        except (KeyboardInterrupt, EOFError):
            print("\n" + Colors.DIM + "Operazione annullata." + Colors.RESET)
            return False
    else:
        print_info("Operazione annullata dall'utente.")
        return False
