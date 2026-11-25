"""
Heuristic & Rule-Based Natural Language Translation Engine for Git.
Understands common phrasing in Italian and English without requiring any LLM or internet.
"""
import re
from typing import Dict, Optional
from .git_context import GitContext


class MatchResult:
    def __init__(self, command: str, explanation: str, is_destructive: bool = False, confidence: float = 1.0, preview: str = ""):
        self.command = command
        self.explanation = explanation
        self.is_destructive = is_destructive
        self.confidence = confidence
        self.preview = preview

    def to_dict(self) -> Dict:
        return {
            "command": self.command,
            "explanation": self.explanation,
            "is_destructive": self.is_destructive,
            "confidence": self.confidence,
            "preview": self.preview
        }


def extract_quotes(text: str) -> Optional[str]:
    """Extract string within single or double quotes."""
    match = re.search(r'["\']([^"\']+)["\']', text)
    if match:
        return match.group(1).strip()
    return None


def extract_after_keyword(text: str, keywords: list) -> Optional[str]:
    """Extracts text following a specific keyword or preposition."""
    lower = text.lower()
    for kw in keywords:
        pos = lower.find(kw)
        if pos != -1:
            val = text[pos + len(kw):].strip()
            # Clean leading colons, quotes or prepositions
            val = re.sub(r'^[:\s"\']+', '', val)
            val = re.sub(r'["\']+$', '', val)
            if val:
                return val.strip()
    return None


def match_heuristic(query: str, ctx: Optional[GitContext] = None) -> Optional[MatchResult]:
    """
    Evaluates natural language query against pattern rules.
    Returns MatchResult if matched, otherwise None.
    """
    q = query.strip()
    low = q.lower()
    curr_branch = ctx.branch if (ctx and ctx.branch) else "main"

    # --- 1. REPO INIT / CLONE ---
    if re.search(r'\b(inizializza|crea|init)\b.*\b(repo|repository|git)\b', low) or low in ["git init", "inizializza repo"]:
        return MatchResult("git init", "Inizializza un nuovo repository Git in questa cartella.")

    clone_match = re.search(r'\b(clona|scarica|clone)\b\s+(https?://[^\s]+|git@[^\s]+)', q, re.I)
    if clone_match:
        url = clone_match.group(2)
        return MatchResult(f"git clone {url}", f"Clona il repository remoto da '{url}'.")

    # --- 2. STATUS & INSPECTION ---
    if re.search(r'\b(cosa ho (modificato|cambiato|toccato)|quali file sono (cambiati|modificati)|stato|status|situazione|mostra (lo )?stato)\b', low):
        return MatchResult("git status", "Mostra lo stato dell'albero di lavoro e dell'area di staging.")

    # Diff
    if re.search(r'\b(differenz[ae]|diff|cosa è cambiato|guarda le modifiche)\b', low):
        if "staging" in low or "staged" in low or "pront" in low:
            return MatchResult("git diff --staged", "Mostra le differenze già aggiunte all'area di staging.")
        # Specific file diff?
        file_m = re.search(r'\b(di|del file|sul file|per il file)\s+([\w\.\-\/\\_]+)', q, re.I)
        if file_m and not file_m.group(2) in ["staging", "staged"]:
            filename = file_m.group(2)
            return MatchResult(f"git diff {filename}", f"Mostra le modifiche non ancora staged nel file '{filename}'.")
        return MatchResult("git diff", "Mostra le differenze non ancora aggiunte all'area di staging.")

    # History & Logs
    if re.search(r'\b(storia|cronologia|log|albero|grafic[oa]|ultimi\s+\d+\s+commit|mostra(mi)?\s+(i\s+)?commit)\b', low):
        count_m = re.search(r'\b(ultimi|ultime|last)\s+(\d+)', low)
        count = count_m.group(2) if count_m else "10"
        if "albero" in low or "grafic" in low or "oneline" in low:
            return MatchResult(
                f"git log --graph --oneline --decorate -n {count}",
                f"Visualizza gli ultimi {count} commit in formato compatto ad albero."
            )
        return MatchResult(
            f"git log -n {count} --oneline",
            f"Mostra la cronologia degli ultimi {count} commit."
        )

    # Blame
    if "chi ha " in low or "blame" in low:
        file_m = re.search(r'\b(file|per|su|di)\s+([\w\.\-\/\\_]+)', q, re.I)
        if file_m:
            fn = file_m.group(2)
            return MatchResult(f"git blame {fn}", f"Mostra riga per riga chi e quando ha modificato '{fn}'.")

    # --- 3. STAGING (ADD / RESTORE STAGED) ---
    # Add all
    if re.search(r'\b(aggiungi tutto|metti tutto in (staging|stage)|traccia tutt[io]|add all|prepara tutto|aggiungi tutte le modifiche)\b', low):
        return MatchResult("git add -A", "Aggiunge tutte le modifiche (nuovi file, modifiche, eliminazioni) all'area di staging.")

    # Remove from stage (unstage)
    if re.search(r'\b(togli|rimuovi|unstage|elimina)\b.*\b(dallo staging|dall\'area di stage|dallo stage)\b', low) or \
       re.search(r'\b(non tracciare|togli dallo staging)\b', low):
        # Look for filename before "dallo staging"
        file_before = re.search(r'\b(?:togli|rimuovi|unstage)\s+(?:il\s+file\s+)?([a-zA-Z0-9_\-\.\/]+)\s+dallo\s+stag', q, re.I)
        if file_before:
            fn = file_before.group(1).strip()
            if fn.lower() not in ["tutto", "tutti", "all"]:
                return MatchResult(f"git restore --staged {fn}", f"Rimuove '{fn}' dall'area di staging senza cancellare le modifiche.")
        # Look for filename after
        file_after = re.search(r'\b(?:stag(?:ing|e))\s+(?:il\s+file\s+)?([a-zA-Z0-9_\-\.\/]+)$', q, re.I)
        if file_after:
            fn = file_after.group(1).strip()
            if fn.lower() not in ["tutto", "tutti", "all"]:
                return MatchResult(f"git restore --staged {fn}", f"Rimuove '{fn}' dall'area di staging senza cancellare le modifiche.")
        return MatchResult("git restore --staged .", "Rimuove tutti i file dall'area di staging senza cancellare le modifiche.")

    # Add specific file
    add_match = re.search(r'\b(aggiungi|metti in staging|traccia|prepara|stage)\s+(?:il\s+file\s+)?([a-zA-Z0-9_\-\.\/]+)', q, re.I)
    if add_match:
        fn = add_match.group(2).strip()
        if fn.lower() not in ["tutto", "tutti", "tutte", "all", "staging", "stage", "commit"]:
            return MatchResult(f"git add {fn}", f"Aggiunge '{fn}' all'area di staging.")

    # --- 4. COMMIT ---
    # Detailed / Descriptive commit with user and file metadata
    if re.search(r'\b(descrivi|dettagliat[oa]|descrittiv[oa])\b.*\b(commit|committa)\b', low) or \
       re.search(r'\b(commit|committa)\b.*\b(con (tutte le )?info|con (tutte le )?informazion|dettagliat|descrittiv)\b', low) or \
       re.search(r'\b(descrivi\s+(un\s+|il\s+)?commit)\b', low):
        try:
            from naturalgit.describe_commit import generate_detailed_commit
        except ImportError:
            from .describe_commit import generate_detailed_commit
        title = extract_quotes(q) or extract_after_keyword(q, ["chiamato", "titolo", "con titolo", "dicendo"])
        if ctx and ctx.is_repo:
            cmd, expl, full_msg = generate_detailed_commit(ctx, title=title, add_all=True)
            return MatchResult(cmd, expl, preview=full_msg)
        else:
            return MatchResult(
                'git add -A && git commit -m "Commit descrittivo con info utente e file"',
                "Crea un commit descrittivo con tutte le informazioni dei file e dell'utente."
            )

    # Combined: add all and commit with message
    if re.search(r'\b(committa tutto|aggiungi tutto e (fai )?commit|salva tutto con|commit tutto)\b', low):
        msg = extract_quotes(q)
        if not msg:
            msg = extract_after_keyword(q, ["dicendo", "messaggio", "con messaggio", "con", "commit"])
        if not msg:
            msg = "Aggiornamento modifiche"
        return MatchResult(
            f'git add -A && git commit -m "{msg}"',
            f'Aggiunge tutti i file modificati e crea un commit con messaggio: "{msg}".'
        )

    # Standard commit
    if re.search(r'\b(fai (un )?commit|committa|crea (un )?commit|salva con messaggio)\b', low):
        msg = extract_quotes(q)
        if not msg:
            msg = extract_after_keyword(q, ["dicendo", "messaggio", "con messaggio", "con: ", "con "])
        if not msg:
            msg = "Aggiornamento codice"
        return MatchResult(
            f'git commit -m "{msg}"',
            f'Crea un commit con i file in staging con messaggio: "{msg}".'
        )

    # Amend
    if re.search(r'\b(amend|modifica l\'ultimo commit|correggi l\'ultimo commit|cambia messaggio dell\'ultimo commit)\b', low):
        msg = extract_quotes(q)
        if msg:
            return MatchResult(f'git commit --amend -m "{msg}"', f'Aggiorna l\'ultimo commit cambiando il messaggio in: "{msg}".')
        return MatchResult("git commit --amend", "Modifica l'ultimo commit aggiungendo le modifiche in staging o cambiando il messaggio.")

    # --- 5. BRANCHING ---
    # List branches
    if re.search(r'\b(elenca|mostra|quali|lista)\b.*\b(branch|rami)\b', low) or low in ["branch", "branches", "lista branch"]:
        return MatchResult("git branch -a", "Elenca tutti i branch locali e remoti.")

    # Delete branch (force)
    del_force = re.search(r'\b(forza (la )?cancellazione|elimina forzatamente|elimina forzato|cancella forzatamente)\b.*\bbranch\b\s+([a-zA-Z0-9_\-\/]+)', q, re.I)
    if del_force:
        b_name = del_force.group(3)
        return MatchResult(f"git branch -D {b_name}", f"Elimina forzatamente il branch locale '{b_name}'.", is_destructive=True)

    # Delete branch (normal)
    del_match = re.search(r'\b(elimina|cancella|rimuovi|delete)\s+(?:il\s+)?branch\s+([a-zA-Z0-9_\-\/]+)', q, re.I)
    if del_match:
        b_name = del_match.group(2)
        return MatchResult(f"git branch -d {b_name}", f"Elimina in sicurezza il branch locale '{b_name}'.", is_destructive=True)

    # Create & checkout branch
    create_switch = re.search(r'\b(crea e (spostati|vai)|nuovo branch e vai|crea nuovo branch.*e vai)\b.*\b(sul|su|chiamato)?\s+([a-zA-Z0-9_\-\/]+)', q, re.I)
    if create_switch:
        b_name = create_switch.group(4)
        return MatchResult(f"git checkout -b {b_name}", f"Crea il nuovo branch '{b_name}' e ci si sposta sopra.")

    # Switch / Checkout existing branch
    switch_match = re.search(r'\b(spostati|passa|vai|cambia branch|checkout|switch)\s+(?:sul|al|a|su|nel)?\s*(?:branch\s+)?([a-zA-Z0-9_\-\/]+)$', q, re.I)
    if switch_match:
        b_name = switch_match.group(2)
        if b_name.lower() not in ["tutto", "tutti", "nuovo", "branch"]:
            return MatchResult(f"git checkout {b_name}", f"Passa al branch '{b_name}'.")

    # Create new branch
    create_branch = re.search(r'\b(crea|nuovo|genera)\s+(?:un\s+)?(?:nuovo\s+)?branch\s+(?:chiamato\s+)?([a-zA-Z0-9_\-\/]+)', q, re.I)
    if create_branch:
        b_name = create_branch.group(2)
        return MatchResult(f"git branch {b_name}", f"Crea il nuovo branch locale '{b_name}'.")

    # --- 6. SYNC (PUSH / PULL / FETCH / REMOTE) ---
    # Remotes
    if re.search(r'\b(mostra|elenca|quali)\b.*\b(remote|remoti|server)\b', low):
        return MatchResult("git remote -v", "Elenca gli indirizzi dei repository remoti configurati.")

    # Push force
    if re.search(r'\b(forza il push|push forzato|invia forzatamente|push --force)\b', low):
        return MatchResult(f"git push --force-with-lease origin {curr_branch}", f"Invia forzatamente i commit al branch remoto '{curr_branch}' (con protezione lease).", is_destructive=True)

    # Push standard
    if re.search(r'\b(push|invia|carica|pubblica)\b.*\b(commit|modifiche|sul server|remoto|su origin|online)?\b', low):
        if not re.search(r'\b(pull|scarica|prendi)\b', low):
            return MatchResult(f"git push origin {curr_branch}", f"Invia i commit locali al server remoto sul branch '{curr_branch}'.")

    # Pull standard
    if re.search(r'\b(pull|scarica|aggiorna|prendi|sincronizza)\b.*\b(dal server|novità|ultime modifiche|repository|da origin)?\b', low):
        return MatchResult("git pull", "Scarica e integra le ultime modifiche dal server remoto nel branch corrente.")

    # Fetch
    if re.search(r'\b(fetch|scarica solo informazioni|controlla novità senza applicarle)\b', low):
        return MatchResult("git fetch --all", "Scarica tutti i riferimenti e branch dal server senza modificare i file locali.")

    # --- 7. STASH ---
    if re.search(r'\b(metti da parte|nascondi|salva temporaneamente|metti in stash|crea stash)\b', low):
        return MatchResult("git stash", "Mette al sicuro temporaneamente tutte le modifiche non committate nell'area di stash.")

    if re.search(r'\b(riprendi|recupera|applica|riapplica|ripristina)\b.*\b(stash|modifiche messe da parte|modifiche temporanee)\b', low) or low in ["stash pop", "pop stash"]:
        return MatchResult("git stash pop", "Recupera l'ultimo gruppo di modifiche salvate nello stash e lo rimuove dallo stash.")

    if re.search(r'\b(elenca|elenco|mostra|lista)\b.*\b(stash)\b', low):
        return MatchResult("git stash list", "Mostra l'elenco di tutte le modifiche salvate nello stash.")

    if re.search(r'\b(cancella|svuota|butta|drop)\b.*\b(stash)\b', low):
        return MatchResult("git stash drop", "Elimina l'ultimo salvataggio dallo stash.", is_destructive=True)

    # --- 8. UNDO / RESET / RESTORE ---
    # Discard single file changes
    restore_file = re.search(r'\b(annulla|scarta|ripristina|cancella)\b.*\b(modifiche al file|il file)\s+([a-zA-Z0-9_\-\.\/]+)', q, re.I)
    if restore_file:
        fn = restore_file.group(3)
        return MatchResult(f"git restore {fn}", f"Scarta tutte le modifiche non committate al file '{fn}'.", is_destructive=True)

    # Discard all unstaged changes
    if re.search(r'\b(annulla tutte le modifiche|scarta tutte le modifiche|torna a com\'era prima|butta tutte le modifiche)\b', low):
        return MatchResult("git restore . && git clean -fd", "Ripristina tutti i file tracciati e rimuove i nuovi file non tracciati.", is_destructive=True)

    # Soft reset (undo last commit, keep changes)
    if re.search(r'\b(annulla l\'ultimo commit|disfa l\'ultimo commit)\b.*\b(tenendo|tieni|lasciando|mantenendo)\b', low) or \
       re.search(r'\b(soft reset|togli l\'ultimo commit lasciando)\b', low):
        return MatchResult("git reset --soft HEAD~1", "Annulla l'ultimo commit lasciando tutti i file modificati nell'area di staging.")

    # Hard reset (undo last commit and discard changes)
    if re.search(r'\b(cancella l\'ultimo commit|hard reset|elimina l\'ultimo commit e le modifiche)\b', low):
        return MatchResult("git reset --hard HEAD~1", "Elimina completamente l'ultimo commit e cancella tutte le relative modifiche.", is_destructive=True)

    # Revert
    if re.search(r'\b(revert|crea commit inverso|annulla con un nuovo commit)\b', low):
        return MatchResult("git revert HEAD", "Crea un nuovo commit che applica l'esatto opposto dell'ultimo commit.")

    # --- 9. MERGE & REBASE ---
    # Abort merge
    if re.search(r'\b(annulla|abortisci|cancella)\b.*\bmerge\b', low):
        return MatchResult("git merge --abort", "Interrompe il merge in corso e ripristina la situazione precedente.")

    # Merge branch
    merge_m = re.search(r'\b(unisci|fai il merge di|merge con|incorpora)\s+(?:il\s+branch\s+)?([a-zA-Z0-9_\-\/]+)', q, re.I)
    if merge_m:
        target_b = merge_m.group(2)
        return MatchResult(f"git merge {target_b}", f"Unisce le modifiche del branch '{target_b}' nel branch corrente '{curr_branch}'.")

    # Rebase
    rebase_m = re.search(r'\b(fai (il )?rebase su|rebase con|riallinea su)\s+([a-zA-Z0-9_\-\/]+)', q, re.I)
    if rebase_m:
        target_b = rebase_m.group(3)
        return MatchResult(f"git rebase {target_b}", f"Riapplica i commit correnti in cima al branch '{target_b}'.")

    # --- 10. TAGS ---
    tag_m = re.search(r'\b(crea|assegna|nuovo)\s+tag\s+([a-zA-Z0-9_\-\.]+)', q, re.I)
    if tag_m:
        tag_name = tag_m.group(2)
        return MatchResult(f"git tag {tag_name}", f"Crea il tag '{tag_name}' sul commit corrente.")

    if re.search(r'\b(elenca|mostra|lista)\b.*\btag\b', low):
        return MatchResult("git tag -l", "Elenca tutti i tag disponibili.")

    return None
