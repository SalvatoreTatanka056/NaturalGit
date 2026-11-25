"""
Main CLI Interface and Interactive REPL for NaturalGit.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Ensure the root package folder is in sys.path for direct script invocation
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

try:
    from naturalgit.ui import (
        Colors, header, print_status, print_error, print_warning,
        print_info, print_success
    )
    from naturalgit.git_context import GitContext
    from naturalgit.config import load_config, save_config, DEFAULT_CONFIG
    from naturalgit.heuristic_engine import match_heuristic
    from naturalgit.llm_engine import query_llm
    from naturalgit.executor import execute_proposal
except ImportError:
    from .ui import (
        Colors, header, print_status, print_error, print_warning,
        print_info, print_success
    )
    from .git_context import GitContext
    from .config import load_config, save_config, DEFAULT_CONFIG
    from .heuristic_engine import match_heuristic
    from .llm_engine import query_llm
    from .executor import execute_proposal


def process_query(
    query: str,
    ctx: GitContext,
    config: dict,
    auto_confirm: bool = False,
    dry_run: bool = False
) -> bool:
    """Processes a natural language query using heuristics and/or LLM."""
    query = query.strip()
    if not query:
        return False

    # Check if this is already an explicit git command (e.g., user typed "git status" or "git checkout main")
    if query.startswith("git "):
        return execute_proposal(
            command=query,
            explanation="Comando Git diretto specificato dall'utente.",
            engine_name="Comando diretto",
            is_destructive=False,
            auto_confirm=auto_confirm,
            dry_run=dry_run
        )

    # 1. Try local heuristic pattern engine first (instant, works offline)
    heuristic_res = match_heuristic(query, ctx)
    if heuristic_res:
        return execute_proposal(
            command=heuristic_res.command,
            explanation=heuristic_res.explanation,
            engine_name="Regole locali",
            is_destructive=heuristic_res.is_destructive,
            auto_confirm=auto_confirm,
            dry_run=dry_run,
            preview=heuristic_res.preview
        )

    # 2. If heuristics did not match, try LLM (Ollama, Gemini, OpenAI)
    llm_res = query_llm(query, ctx, config)
    if llm_res:
        provider_name = config.get("provider", "LLM").upper()
        return execute_proposal(
            command=llm_res.command,
            explanation=llm_res.explanation,
            engine_name=f"Intelligenza Artificiale ({provider_name})",
            is_destructive=llm_res.is_destructive,
            auto_confirm=auto_confirm,
            dry_run=dry_run
        )

    # 3. If neither matched
    print_warning(f"Non sono riuscito a interpretare: '{query}'")
    print_info("Suggerimento: prova a riformulare (es: 'crea branch test', 'aggiungi tutto e committa', 'mostra ultimi commit').")
    if config.get("provider") == "heuristic" or (not config.get("gemini_api_key") and not config.get("openai_api_key")):
        print(f" {Colors.DIM}Puoi configurare un LLM (Ollama, Gemini, OpenAI) con: ngit config --provider gemini --api-key <CHIAVE>{Colors.RESET}\n")
    return False


def handle_config_command(args: argparse.Namespace, config: dict):
    """Handles 'ngit config' management."""
    if args.show:
        print(f"\n{Colors.BOLD}{Colors.CYAN}--- Configurazione NaturalGit ---{Colors.RESET}")
        for k, v in config.items():
            if "api_key" in k and v:
                display_val = v[:4] + "..." + v[-4:] if len(v) > 8 else "***"
            else:
                display_val = str(v)
            print(f"  {Colors.BOLD}{k}:{Colors.RESET} {display_val}")
        print()
        return

    if args.reset:
        save_config(DEFAULT_CONFIG)
        print_success("Configurazione ripristinata ai valori predefiniti.")
        return

    modified = False
    if args.provider:
        config["provider"] = args.provider.lower()
        modified = True
    if args.api_key:
        if config.get("provider") == "gemini":
            config["gemini_api_key"] = args.api_key
        else:
            config["openai_api_key"] = args.api_key
        modified = True
    if args.model:
        if config.get("provider") == "gemini":
            config["gemini_model"] = args.model
        elif config.get("provider") == "ollama":
            config["ollama_model"] = args.model
        else:
            config["openai_model"] = args.model
        modified = True
    if args.ollama_url:
        config["ollama_url"] = args.ollama_url
        modified = True

    if modified:
        if save_config(config):
            print_success("Configurazione salvata con successo!")
        else:
            print_error("Errore durante il salvataggio della configurazione.")
    else:
        print_info("Usa 'ngit config --show' per visualizzare i parametri o '--help' per opzioni.")


def run_repl(ctx: GitContext, config: dict):
    """Runs interactive NaturalGit shell."""
    header()
    print(f"  {Colors.DIM}Modalità interattiva. Digita comandi in linguaggio naturale.{Colors.RESET}")
    print(f"  {Colors.DIM}Digita '{Colors.WHITE}esci{Colors.DIM}' o '{Colors.WHITE}exit{Colors.DIM}' per uscire, '{Colors.WHITE}aiuto{Colors.DIM}' per esempi.{Colors.RESET}\n")

    while True:
        # Refresh git context on each prompt
        ctx._refresh()
        branch_str = f"({ctx.branch})" if ctx.branch else "(no repo)"
        prompt_symbol = f"{Colors.BOLD}{Colors.BRIGHT_MAGENTA}ngit {branch_str}{Colors.RESET}> "

        try:
            user_input = input(prompt_symbol).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n" + Colors.DIM + "Arrivederci!" + Colors.RESET)
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "esci", "quit", "q"]:
            print(f"{Colors.DIM}Arrivederci! 👋{Colors.RESET}")
            break

        if user_input.lower() in ["clear", "cls"]:
            import os
            os.system("cls" if os.name == "nt" else "clear")
            header()
            continue

        if user_input.lower() in ["aiuto", "help"]:
            print(f"""
{Colors.BOLD}Esempi di comandi che puoi digitare:{Colors.RESET}
  - "mostrami lo stato dei file"
  - "mostra cosa è cambiato"
  - "aggiungi tutto"
  - "committa dicendo 'fix problema con login'"
  - "committa tutto con messaggio 'aggiornamento generale'"
  - "crea il branch feature/pagamenti e spostati lì"
  - "elenca tutti i branch"
  - "spostati su main"
  - "invia le modifiche al server" (push)
  - "scarica le novità" (pull)
  - "metti da parte le modifiche temporaneamente" (stash)
  - "recupera le modifiche messe da parte" (stash pop)
  - "mostra gli ultimi 5 commit ad albero"
  - "annulla le modifiche al file main.py"
  - "annulla l'ultimo commit tenendo i file"
""")
            continue

        process_query(user_input, ctx, config)
        print()


def main(argv: Optional[List[str]] = None):
    if argv is None:
        argv = sys.argv[1:]

    config = load_config()
    ctx = GitContext()

    # Check for config subcommand explicitly first
    if argv and argv[0] == "config":
        config_parser = argparse.ArgumentParser(
            prog="ngit config",
            description="Configura NaturalGit (LLM, provider, chiavi)"
        )
        config_parser.add_argument("--show", action="store_true", help="Mostra la configurazione corrente")
        config_parser.add_argument("--provider", choices=["auto", "ollama", "gemini", "openai", "heuristic"], help="Seleziona il provider")
        config_parser.add_argument("--api-key", help="Imposta la chiave API per il provider selezionato")
        config_parser.add_argument("--model", help="Imposta il modello (es: llama3, gemini-1.5-flash, gpt-4o-mini)")
        config_parser.add_argument("--ollama-url", help="URL dell'endpoint Ollama (default: http://localhost:11434)")
        config_parser.add_argument("--reset", action="store_true", help="Ripristina le impostazioni predefinite")
        config_args = config_parser.parse_args(argv[1:])
        handle_config_command(config_args, config)
        return

    # Standard query parser
    parser = argparse.ArgumentParser(
        prog="ngit",
        description="🌿 NaturalGit: Esegui operazioni Git usando il linguaggio naturale.",
        epilog="Esempio: ngit 'committa tutto dicendo fix bug login'"
    )

    parser.add_argument("query", nargs="*", help="Istruzione in linguaggio naturale (in italiano o inglese).")
    parser.add_argument("-dc", "--describe-commit", action="store_true", help="Crea un commit descrittivo con tutti i metadati dell'utente e dei file.")
    parser.add_argument("-y", "--yes", action="store_true", help="Esegui automaticamente senza chiedere conferma.")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Mostra il comando identificato senza eseguirlo.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Avvia la shell interattiva REPL.")

    args = parser.parse_args(argv)

    if args.describe_commit:
        try:
            from naturalgit.describe_commit import generate_detailed_commit
        except ImportError:
            from .describe_commit import generate_detailed_commit
        header()
        if not ctx.is_repo:
            print_error("Operazione non possibile: non ti trovi all'interno di un repository Git.")
            return
        print_status(ctx.branch, ctx.clean, ctx.staged_count, ctx.unstaged_count, ctx.untracked_count)
        custom_title = " ".join(args.query) if args.query else None
        cmd, expl, full_msg = generate_detailed_commit(ctx, title=custom_title, add_all=True)
        execute_proposal(
            command=cmd,
            explanation=expl,
            engine_name="Commit Descrittivo",
            is_destructive=False,
            auto_confirm=args.yes,
            dry_run=args.dry_run,
            preview=full_msg
        )
        return

    if args.interactive or not args.query:
        run_repl(ctx, config)
        return

    query_text = " ".join(args.query)
    header()
    if ctx.is_repo:
        print_status(ctx.branch, ctx.clean, ctx.staged_count, ctx.unstaged_count, ctx.untracked_count)
    else:
        print_warning("Non ti trovi all'interno di un repository Git (puoi dire 'inizializza repo' per crearlo).")
    
    process_query(query_text, ctx, config, auto_confirm=args.yes, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
