"""
Main CLI Interface and Interactive REPL for NaturalGit.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

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
    from naturalgit.memory import get_memory
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
    from .memory import get_memory


def process_query(
    query: str,
    ctx: GitContext,
    config: dict,
    auto_confirm: bool = False,
    dry_run: bool = False,
    use_local: bool = False
) -> bool:
    """
    Multi-stage high-speed translation pipeline:
    1. Direct Git commands (< 1ms)
    2. Local llmasm C/Assembly Engine (if --local or provider=local)
    3. Local Heuristic Engine (< 5ms)
    4. User-Trained Rules / Few-Shot Learning (< 2ms)
    5. Local AI Response Cache (< 2ms)
    6. Remote / Local LLM (Gemini Flash / Ollama) -> automatically cached for future instant hits!
    """
    query = query.strip()
    if not query:
        return False

    memory = get_memory()

    # 0. Check if already an explicit git command (e.g. "git status")
    if query.startswith("git "):
        return execute_proposal(
            command=query,
            explanation="Comando Git diretto specificato dall'utente.",
            engine_name="Comando diretto",
            is_destructive=False,
            auto_confirm=auto_confirm,
            dry_run=dry_run,
            original_query=query
        )

    provider = config.get("provider", "auto").lower()
    has_local_model = bool(config.get("local_model_path"))

    # 1. Check if user requested local llmasm model (--local or provider=local or auto with local_model_path)
    if use_local or provider in ["local", "llmasm"] or (provider == "auto" and has_local_model):
        llm_res = query_llm(query, ctx, config, force_local=True)
        if llm_res:
            backend_name = config.get("local_backend", "auto").upper()
            return execute_proposal(
                command=llm_res.command,
                explanation=llm_res.explanation,
                engine_name=f"Motore Locale llmasm ({backend_name})",
                is_destructive=llm_res.is_destructive,
                auto_confirm=auto_confirm,
                dry_run=dry_run,
                original_query=query
            )
        elif provider in ["local", "llmasm"]:
            # If user explicitly configured provider=local, try offline heuristic fallback before giving up
            heur_res = match_heuristic(query, ctx)
            if heur_res:
                return execute_proposal(
                    command=heur_res.command,
                    explanation=heur_res.explanation,
                    engine_name="Motore Euristico Locale (Offline Fallback)",
                    is_destructive=heur_res.is_destructive,
                    auto_confirm=auto_confirm,
                    dry_run=dry_run,
                    original_query=query
                )
            print_warning("Provider impostato su 'local' ma l'esecuzione con llmasm non è riuscita.")
            print_info("Verifica il percorso del modello con 'ngit config --show'.")
            print_info("Se vuoi passare a un altro provider: 'ngit config --provider gemini'")
            return False
        else:
            print_warning("Impossibile caricare il modello locale llmasm.")
            print_info("Continuo con i provider cloud configurati...\n")

    provider = config.get("provider", "auto").lower()
    has_gemini = bool(config.get("gemini_api_key"))
    has_openai = bool(config.get("openai_api_key"))
    has_ollama = bool(provider == "ollama")

    # If user selected an AI provider (e.g. Gemini), AI IS THE PRIMARY AGENT for any sentence!
    always_ai = (provider in ["gemini", "openai", "ollama"]) or (provider == "auto" and (has_gemini or has_openai))

    if always_ai:
        # Check cache (< 1ms) so repeated queries don't wait on network
        cached = memory.get_from_cache(query)
        if cached:
            return execute_proposal(
                command=cached["command"],
                explanation=cached["explanation"],
                engine_name=cached["source"],
                is_destructive=cached.get("is_destructive", False),
                auto_confirm=auto_confirm,
                dry_run=dry_run,
                original_query=query
            )

        # Primary: Run Gemini AI Agent!
        llm_res = query_llm(query, ctx, config)
        if llm_res:
            memory.store_in_cache(query, llm_res.command, llm_res.explanation, llm_res.is_destructive)
            agent_name = "Gemini Flash" if provider == "gemini" else provider.upper()
            return execute_proposal(
                command=llm_res.command,
                explanation=llm_res.explanation,
                engine_name=f"Agente AI ({agent_name})",
                is_destructive=llm_res.is_destructive,
                auto_confirm=auto_confirm,
                dry_run=dry_run,
                original_query=query
            )

        # Offline / Fallback: If AI is unreachable, use local heuristic engine
        heuristic_res = match_heuristic(query, ctx)
        if heuristic_res:
            return execute_proposal(
                command=heuristic_res.command,
                explanation=heuristic_res.explanation,
                engine_name="Regole locali (fallback offline)",
                is_destructive=heuristic_res.is_destructive,
                auto_confirm=auto_confirm,
                dry_run=dry_run,
                preview=heuristic_res.preview,
                original_query=query
            )
    else:
        # Standard heuristic first pipeline
        heuristic_res = match_heuristic(query, ctx)
        if heuristic_res:
            return execute_proposal(
                command=heuristic_res.command,
                explanation=heuristic_res.explanation,
                engine_name="Regole locali (ultra-veloce)",
                is_destructive=heuristic_res.is_destructive,
                auto_confirm=auto_confirm,
                dry_run=dry_run,
                preview=heuristic_res.preview,
                original_query=query
            )

        learned = memory.match_learned(query)
        if learned:
            return execute_proposal(
                command=learned["command"],
                explanation=learned["explanation"],
                engine_name=learned["source"],
                is_destructive=learned.get("is_destructive", False),
                auto_confirm=auto_confirm,
                dry_run=dry_run,
                original_query=query
            )

    # 3. Check local AI response cache (< 2ms)
    cached = memory.get_from_cache(query)
    if cached:
        return execute_proposal(
            command=cached["command"],
            explanation=cached["explanation"],
            engine_name=cached["source"],
            is_destructive=cached.get("is_destructive", False),
            auto_confirm=auto_confirm,
            dry_run=dry_run,
            original_query=query
        )

    # 4. Query LLM (Gemini Flash-Lite / Ollama / OpenAI)
    llm_res = query_llm(query, ctx, config)
    if llm_res:
        # Automatically cache this resolution so future queries are sub-millisecond!
        memory.store_in_cache(query, llm_res.command, llm_res.explanation, llm_res.is_destructive)
        provider_name = config.get("provider", "LLM").upper()
        return execute_proposal(
            command=llm_res.command,
            explanation=llm_res.explanation,
            engine_name=f"Intelligenza Artificiale ({provider_name})",
            is_destructive=llm_res.is_destructive,
            auto_confirm=auto_confirm,
            dry_run=dry_run,
            original_query=query
        )

    # 5. Fallback if neither matched
    print_warning(f"Non sono riuscito a interpretare: '{query}'")
    print_info("Suggerimento: prova a riformulare oppure insegna a NaturalGit questo comando con:")
    print(f"  {Colors.BOLD}{Colors.CYAN}ngit learn \"{query}\" \"comando git esatto\"{Colors.RESET}\n")
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
    if getattr(args, "gemini_model", None):
        config["gemini_model"] = args.gemini_model
        modified = True
    if getattr(args, "ollama_model", None):
        config["ollama_model"] = args.ollama_model
        modified = True
    if getattr(args, "openai_model", None):
        config["openai_model"] = args.openai_model
        modified = True
    if getattr(args, "local_model", None):
        clean_path = args.local_model.strip().strip('"').strip("'")
        config["local_model_path"] = clean_path
        config["provider"] = "local"
        modified = True
        print_info(f"Provider impostato automaticamente su 'local' (modello: {clean_path})")
    if getattr(args, "local_backend", None):
        config["local_backend"] = args.local_backend
        modified = True
    if getattr(args, "local_threads", None):
        config["local_threads"] = args.local_threads
        modified = True
    if args.ollama_url:
        config["ollama_url"] = args.ollama_url
        modified = True
    if args.auto_confirm is not None:
        config["auto_confirm"] = args.auto_confirm.lower() in ["true", "1", "yes", "si"]
        modified = True

    if modified:
        if save_config(config):
            print_success("Configurazione salvata con successo!")
        else:
            print_error("Errore durante il salvataggio della configurazione.")
    else:
        print_info("Usa 'ngit config --show' per visualizzare i parametri o '--help' per opzioni.")


def handle_learn_command(args: List[str]):
    """Handles teaching a new custom command: ngit learn <frase> <comando> [spiegazione]."""
    if len(args) < 2:
        print_error("Uso: ngit learn \"<frase in linguaggio naturale>\" \"<comando git da eseguire>\" [\"<spiegazione>\"]")
        print_info("Esempio: ngit learn \"mostra i nomi dei file modificati\" \"git status -s\" \"Mostra solo i nomi dei file\"")
        return

    phrase = args[0]
    command = args[1]
    explanation = args[2] if len(args) > 2 else f"Comando appreso per: '{phrase}'"

    memory = get_memory()
    if memory.learn(phrase, command, explanation):
        print_success(f"Regola appresa con successo!")
        print(f"  {Colors.DIM}Frase:{Colors.RESET}   \"{phrase}\"")
        print(f"  {Colors.DIM}Comando:{Colors.RESET} {Colors.GREEN}{command}{Colors.RESET}")
        print(f"  {Colors.DIM}Effetto:{Colors.RESET} Ora risponderà istantaneamente (<2ms) senza passare per l'AI!\n")
    else:
        print_error("Errore durante il salvataggio della regola.")


def handle_rules_command(argv: List[str]):
    """Manages custom learned rules."""
    memory = get_memory()
    if "--delete" in argv:
        idx = argv.index("--delete")
        if idx + 1 < len(argv):
            target = argv[idx + 1]
            if memory.delete_learned(target):
                print_success(f"Regola per '{target}' eliminata.")
            else:
                print_warning(f"Nessuna regola trovata per '{target}'.")
            return
        else:
            print_error("Specifica la frase della regola da eliminare: ngit rules --delete \"<frase>\"")
            return

    rules = memory.list_learned()
    if not rules:
        print_info("Nessuna regola personalizzata salvata.")
        print(f"  {Colors.DIM}Puoi insegnarne una con: ngit learn \"<frase>\" \"<comando>\"{Colors.RESET}\n")
        return

    print(f"\n{Colors.BOLD}{Colors.CYAN}--- Regole Personalizzate Apprese ({len(rules)}) ---{Colors.RESET}")
    for i, r in enumerate(rules, 1):
        print(f" {Colors.BOLD}{i}.{Colors.RESET} \"{r['phrase']}\" -> {Colors.GREEN}{r['command']}{Colors.RESET}")
        if r.get("explanation"):
            print(f"    {Colors.DIM}💡 {r['explanation']}{Colors.RESET}")
    print()


def handle_cache_command(argv: List[str]):
    """Manages AI response cache."""
    memory = get_memory()
    if "--clear" in argv:
        memory.clear_cache()
        print_success("Cache locale delle risposte AI svuotata.")
        return
    count = len(memory.ai_cache)
    print_info(f"Elementi memorizzati nella cache locale AI: {count}")
    print(f"  {Colors.DIM}Usa 'ngit cache --clear' per svuotarla.{Colors.RESET}")


def run_repl(ctx: GitContext, config: dict):
    """Runs interactive NaturalGit shell."""
    header()
    print(f"  {Colors.DIM}Modalità interattiva. Digita comandi in linguaggio naturale.{Colors.RESET}")
    print(f"  {Colors.DIM}Digita '{Colors.WHITE}esci{Colors.DIM}' per uscire, '{Colors.WHITE}regole{Colors.DIM}' per vedere quelle apprese, '{Colors.WHITE}aiuto{Colors.DIM}' per esempi.{Colors.RESET}\n")

    while True:
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

        if user_input.lower() == "regole":
            handle_rules_command([])
            continue

        if user_input.lower() in ["aiuto", "help"]:
            print(f"""
{Colors.BOLD}Esempi di comandi che puoi digitare:{Colors.RESET}
  - "mostrami lo stato dei file" / "nomi dei file modificati"
  - "mostra cosa è cambiato"
  - "aggiungi tutto"
  - "committa dicendo 'fix login'"
  - "descrivi un commit con tutte le informazioni dei file e dell'utente"
  - "crea il branch feature/pagamenti e spostati lì"
  - "invia le modifiche al server" (push)
  - "scarica le novità" (pull)
  - "metti da parte le modifiche temporaneamente" (stash)
  - "recupera le modifiche messe da parte" (stash pop)
  - "cerca nei commit la parola fix"
  - "mostra gli ultimi 5 commit ad albero"
""")
            continue

        process_query(user_input, ctx, config)
        print()


def print_configuration_help():
    """Prints comprehensive terminal guide for all configurations."""
    print(f"""
{Colors.BOLD}{Colors.CYAN}=== 🌿 GUIDA CONFIGURAZIONI NATURALGIT (ngit) ==={Colors.RESET}

{Colors.BOLD}1. VISUALIZZARE CONFIGURAZIONE ATTUALE:{Colors.RESET}
   ngit config --show

{Colors.BOLD}2. SCEGLIERE UN PROVIDER AI (--provider):{Colors.RESET}
   ngit config --provider local        # Motore locale llmasm in C (100% offline, zero internet)
   ngit config --provider gemini       # Google Gemini (ultra-rapido, default: flash-lite)
   ngit config --provider ollama       # Ollama locale (llama3, qwen2.5-coder)
   ngit config --provider openai       # OpenAI o compatibili (vLLM, LM Studio)
   ngit config --provider heuristic    # Regole euristiche offline (<5ms, senza AI)
   ngit config --provider auto         # Selezione automatica intelligente

{Colors.BOLD}3. MOTORE LOCALE C / ASSEMBLY (llmasm):{Colors.RESET}
   ngit config --local-model "C:\\percorso\\modello.asmbin"   # Imposta modello e attiva provider local
   ngit config --local-backend c                             # Backend C (AVX2 DLL nativa) o numpy
   ngit config --local-threads 8                             # Numero di thread CPU

{Colors.BOLD}4. MODELLI CLOUD ED OLLAMA:{Colors.RESET}
   ngit config --api-key "CHIAVE_API"                        # Salva la chiave API (Gemini/OpenAI)
   ngit config --model gemini-flash-lite-latest              # Modello per il provider attivo
   ngit config --ollama-url http://localhost:11434           # URL server Ollama

{Colors.BOLD}5. ESECUZIONE AUTOMATICA SENZA PROMPT:{Colors.RESET}
   ngit config --auto-confirm true                           # Esegue senza chiedere [Si/no]
   ngit config --auto-confirm false                          # Chiede sempre conferma (consigliato)

{Colors.BOLD}6. FLAG DI ESECUZIONE AL VOLO:{Colors.RESET}
   ngit -l "frase"                     # Forza motore locale llmasm per questo comando
   ngit -l -j 8 "frase"                # Forza motore locale con 8 thread CPU
   ngit -y "frase"                     # Esegue immediatamente senza prompt
   ngit -d "frase"                     # Dry-run: simula senza toccare il repository
   ngit -dc "titolo opzionale"         # Commit descrittivo completo con metadati utente/file
   ngit -i                             # Avvia la shell interattiva REPL

{Colors.BOLD}7. RIPRISTINO VALORI DI FABBRICA:{Colors.RESET}
   ngit config --reset

{Colors.DIM}Consulta la documentazione completa in: GUIDA_CONFIGURAZIONE.md{Colors.RESET}
""")


def main(argv: Optional[List[str]] = None):
    if argv is None:
        argv = sys.argv[1:]

    config = load_config()
    ctx = GitContext()

    # Help command
    if argv and argv[0] in ["help", "aiuto", "--help-config"]:
        print_configuration_help()
        return

    # Explicit subcommands
    if argv and argv[0] == "config":
        if len(argv) > 1 and argv[1] in ["help", "aiuto"]:
            print_configuration_help()
            return
        config_parser = argparse.ArgumentParser(prog="ngit config", description="Configura NaturalGit")
        config_parser.add_argument("--show", action="store_true", help="Mostra la configurazione corrente")
        config_parser.add_argument("--provider", choices=["auto", "ollama", "gemini", "openai", "heuristic", "local", "llmasm"], help="Seleziona il provider")
        config_parser.add_argument("--api-key", help="Imposta la chiave API")
        config_parser.add_argument("--model", help="Imposta il modello per il provider attivo")
        config_parser.add_argument("--gemini-model", help="Imposta specificamente il modello Gemini (es: gemini-flash-lite-latest)")
        config_parser.add_argument("--ollama-model", help="Imposta specificamente il modello Ollama (es: llama3, qwen2.5-coder:1.5b)")
        config_parser.add_argument("--openai-model", help="Imposta specificamente il modello OpenAI (es: gpt-4o-mini)")
        config_parser.add_argument("--local-model", help="Imposta il percorso del modello .asmbin per il motore locale llmasm")
        config_parser.add_argument("--local-backend", choices=["auto", "c", "numpy"], help="Imposta il backend per llmasm (auto, c o numpy)")
        config_parser.add_argument("--local-threads", type=int, help="Imposta il numero di thread CPU per il motore locale llmasm")
        config_parser.add_argument("--ollama-url", help="URL di Ollama")
        config_parser.add_argument("--auto-confirm", choices=["true", "false"], help="Attiva/disattiva esecuzione automatica senza prompt [true/false]")
        config_parser.add_argument("--reset", action="store_true", help="Ripristina valori predefiniti")
        config_args = config_parser.parse_args(argv[1:])
        if not any(vars(config_args).values()):
            config_args.show = True
        handle_config_command(config_args, config)
        return

    if argv and argv[0] == "learn":
        handle_learn_command(argv[1:])
        return

    if argv and argv[0] == "rules":
        handle_rules_command(argv[1:])
        return

    if argv and argv[0] == "cache":
        handle_cache_command(argv[1:])
        return

    # Standard query parser
    parser = argparse.ArgumentParser(
        prog="ngit",
        description="🌿 NaturalGit: Esegui operazioni Git usando il linguaggio naturale.",
        epilog="Esempio: ngit 'committa tutto dicendo fix bug login'"
    )

    parser.add_argument("query", nargs="*", help="Istruzione in linguaggio naturale (in italiano o inglese).")
    parser.add_argument("-l", "--local", action="store_true", help="Usa il motore locale llmasm (.asmbin / C engine) per l'inferenza.")
    parser.add_argument("-j", "--threads", type=int, default=None, help="Numero di thread CPU per l'inferenza locale (default da config)")
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
    
    if args.threads is not None:
        config["local_threads"] = args.threads

    auto_exec = args.yes or config.get("auto_confirm", False)
    process_query(query_text, ctx, config, auto_confirm=auto_exec, dry_run=args.dry_run, use_local=args.local)


if __name__ == "__main__":
    main()
