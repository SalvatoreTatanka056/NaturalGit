# 🌿 NaturalGit (ngit)

> **Controlla Git parlando in linguaggio naturale (italiano o inglese).**  
> Dimentica la sintassi complessa dei comandi Git: descrivi ciò che vuoi fare e NaturalGit lo trasformerà ed eseguirà per te in sicurezza.

---

## ✨ Caratteristiche Principali

- 🧠 **Doppio Motore di Comprensione**:
  - **Motore Euristico Locale (Offline)**: Funziona istantaneamente **senza connessione internet** e **senza alcuna chiave API**, comprendendo centinaia di frasi idiomatiche in italiano.
  - **Supporto LLM (Intelligenza Artificiale)**: Supporta **Ollama** (100% locale, gratuito e privato), **Google Gemini** e **OpenAI / Groq** per richieste complesse o personalizzate.
- 🛡️ **Sicurezza Prima di Tutto**:
  - Mostra sempre una chiara anteprima del comando Git identificato e una spiegazione di cosa farà.
  - Chiede sempre conferma prima di eseguire (`[S]i / [n]o / [m]odifica manuale`).
  - **Evidenzia in rosso le operazioni distruttive o irreversibili** (es. `reset --hard`, `clean -fd`, `push --force`, `branch -D`).
- 💬 **Modalità Interattiva (REPL)**:
  - Avvia una shell conversazionale digitando semplicemente `ngit`.
- ⚡ **Zero Dipendenze Obbligatorie**:
  - Richiede solo Python 3.8+ (e sfrutta `colorama` per i colori su terminale Windows).

---

## 🚀 Installazione ed Esecuzione

### Metodo 1: Installazione come comando globale `ngit`
Dalla cartella del progetto:
```bash
pip install -e . --no-build-isolation --no-deps
```
Ora puoi digitare ovunque nel terminale:
```bash
ngit "il tuo comando in italiano"
```

### Metodo 2: Esecuzione diretta
Su Windows puoi usare direttamente lo script `.bat`:
```cmd
ngit.bat "mostrami lo stato dei file"
```
oppure con Python:
```cmd
python -m naturalgit.cli "mostrami lo stato dei file"
```

---

## 📖 Esempi di Utilizzo

### 1. Stato e Ispezione
```bash
ngit "cosa ho modificato"
ngit "mostrami lo stato dei file"
ngit "differenze rispetto all'ultimo commit"
ngit "differenze in staging"
ngit "chi ha modificato il file app.py"
```

### 2. Aggiunta e Commit
```bash
ngit "aggiungi tutto"
ngit "metti in staging il file main.py"
ngit "togli il file main.py dallo staging"
ngit "committa dicendo 'fix problema di autenticazione'"
ngit "committa tutto con messaggio 'primo rilascio'"
ngit "modifica l'ultimo commit"

# 🌟 NUOVO: Commit descrittivo con tutte le info di utente e file:
ngit "descrivi un commit con tutte le informazioni dell'utente e dei file"
ngit -dc "Titolo facoltativo del commit"
```

### 3. Gestione Branch
```bash
ngit "crea il branch feature/login"
ngit "crea e spostati sul branch dev"
ngit "spostati sul branch main"
ngit "elenca tutti i branch"
ngit "elimina il branch feature/vecchio"
ngit "elimina forzatamente il branch bugfix"
```

### 4. Sincronizzazione con il Server Remoto
```bash
ngit "invia le modifiche al server"
ngit "scarica le novità"
ngit "scarica solo le informazioni senza applicarle"
ngit "mostrami i server remoti"
```

### 5. Stash (Metti da parte)
```bash
ngit "metti da parte le modifiche temporaneamente"
ngit "recupera le modifiche messe da parte"
ngit "mostra l'elenco degli stash"
ngit "cancella lo stash"
```

### 6. Cronologia e Log
```bash
ngit "mostra gli ultimi 5 commit"
ngit "mostrami la cronologia dei commit ad albero"
```

### 7. Annullamento e Ripristino (Con avvisi di sicurezza ⚠️)
```bash
ngit "annulla le modifiche al file config.py"
ngit "annulla tutte le modifiche non salvate"
ngit "annulla l'ultimo commit ma tieni i file modificati"
ngit "cancella l'ultimo commit"
ngit "crea un commit che annulla l'ultimo"
```

### 8. Merge, Rebase e Tag
```bash
ngit "fai il merge di feature/login"
ngit "abortisci il merge"
ngit "fai il rebase su main"
ngit "crea tag v1.0.0"
ngit "elenca tag"
```

---

## 💬 Modalità Interattiva (REPL)

Basta eseguire `ngit` senza argomenti per entrare nella shell interattiva:

```text
┌────────────────────────────────────────────────────────────┐
│  🌿 NaturalGit - Git con linguaggio naturale v1.0          │
└────────────────────────────────────────────────────────────┘
  Branch: main  |  Stato: 2 modificati, 1 non tracciato

ngit (main)> committa tutto dicendo 'aggiornato modulo auth'

┌─ 🎯 Comando Identificato (Regole locali) ─────────────────┐
│  $ git add -A
│  $ git commit -m "aggiornato modulo auth"
├────────────────────────────────────────────────────────────┤
│  💡 Aggiunge tutti i file modificati e crea un commit con messaggio: "aggiornato modulo auth".
└────────────────────────────────────────────────────────────┘
Eseguire questo comando? [Si/no/modifica]:
```

---

## ⚙️ Configurazione LLM (Opzionale)

Se vuoi potenziare la comprensione con un modello di linguaggio per richieste molto complesse o sfumate, puoi configurare un provider AI:

### Usare Ollama (Locale, 100% Gratuito e Privato)
Installa [Ollama](https://ollama.com/) e avvia ad esempio `llama3` o `qwen2.5-coder`:
```bash
ngit config --provider ollama --model llama3
```

### Usare Google Gemini
```bash
ngit config --provider gemini --api-key LA_TUA_GEMINI_API_KEY
```

### Usare OpenAI o compatibili (Groq, DeepSeek, Together, LM Studio)
```bash
ngit config --provider openai --api-key LA_TUA_OPENAI_API_KEY
```

Visualizza la configurazione attuale:
```bash
ngit config --show
```

---

## 🛠️ Opzioni della CLI

| Opzione | Descrizione |
|---|---|
| `<query>` | L'istruzione in linguaggio naturale (es. `ngit "fai il push"`) |
| `-dc`, `--describe-commit` | Crea un commit descrittivo con tutti i metadati dell'utente e dei file |
| `-y`, `--yes` | Esegue subito il comando senza chiedere conferma (salvo comandi distruttivi) |
| `-d`, `--dry-run` | Mostra solo il comando proposto e la spiegazione senza eseguirlo |
| `-i`, `--interactive` | Avvia la shell interattiva |
| `config` | Gestione delle preferenze e delle chiavi API |
