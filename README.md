# 🌿 NaturalGit (ngit)

> **Controlla Git parlando in linguaggio naturale (italiano o inglese).**  
> Dimentica la sintassi complessa dei comandi Git: descrivi ciò che vuoi fare e NaturalGit lo trasformerà ed eseguirà per te in totale sicurezza, guidato dall'intelligenza artificiale di Google Gemini e da un motore offline istantaneo.

---

## ✨ Caratteristiche Principali

- 🧠 **Agente AI Gemini Primario (Qualsiasi Frase, Zero Frasi Statiche)**:
  - Non sei vincolato a formule fisse o sintassi rigide. Puoi esprimerti con **qualsiasi linguaggio colloquiale, informale, complesso o metaforico** (es: *"fammi un quadro completo della situazione prima che faccia danni"*).
  - Basato sul modello ultra-veloce a bassa latenza **`gemini-flash-lite-latest`** con token ridotti e Time-To-First-Token immediato.
- ⚡ **Cache Locale Sub-Millisecondo (< 2ms)**:
  - Memorizza automaticamente ogni risposta dell'AI. Se ripeti la stessa richiesta (o una frase molto simile), la risposta è istantanea e a costo zero di rete!
- 🎓 **Addestramento su Misura (`ngit learn`)**:
  - Puoi insegnare a NaturalGit nuove associazioni e preferenze personali: l'AI le userà come esempi guida (*Few-Shot Learning*) per adattarsi al tuo stile.
- ⚡ **Motore Euristico Locale Offline (< 5ms)**:
  - Include oltre 50 schemi nativi in italiano e inglese che funzionano all'istante anche senza connessione internet o senza chiavi API.
- 📝 **Commit Descrittivo Intelligente (`-dc`)**:
  - Genera in automatico messaggi di commit professionali contenenti i metadati completi dell'autore (Git/OS/Host), timestamp, branch e il dettaglio analitico di ogni singolo file (+/- righe).
- 🛡️ **Sicurezza Attiva e Protezione Codice**:
  - Mostra sempre il comando prima di lanciarlo con spiegazione in italiano.
  - **Evidenzia in rosso le operazioni distruttive o irreversibili** (es. `reset --hard`, `clean -fd`, `push --force`, `branch -D`).
- 💬 **Modalità Interattiva (REPL)**:
  - Avvia una console conversazionale digitando semplicemente `ngit`.
- 📦 **Multipiattaforma con Installer Pronti**:
  - Pacchetti pronti per Windows (`.exe` autonomo, `install.bat`) e Linux (`.deb`, `install.sh`, `.tar.gz`, `.whl`).

---

## 📦 Pacchetti di Installazione Pronti (`dist/`)

Nella cartella `dist/` trovi i pacchetti precompilati per ogni sistema operativo:

| File | Piattaforma | Descrizione | Dimensione |
|---|---|---|---|
| **`ngit.exe`** | Windows x64 | **Eseguibile autonomo standalone** (funziona anche senza Python installato!) | ~12 MB |
| **`install.bat`** | Windows | **Installer 1-Click** per Windows (installa e registra nel PATH utente) | ~2 KB |
| **`naturalgit_1.0.0_all.deb`** | Linux (Debian / Ubuntu / Mint) | **Pacchetto nativo `.deb`** installabile con `dpkg -i` o `apt` | ~22 KB |
| **`install.sh`** | Linux Universale | **Installer automatico per Linux** (Fedora, Arch, RHEL, CentOS, ecc.) | ~2 KB |
| **`naturalgit-linux.tar.gz`** | Linux Universale | Archivio compresso completo di sorgenti e installer Linux | ~71 KB |
| **`naturalgit-1.0.0-py3-none-any.whl`** | Universale (Win/Linux/Mac) | Pacchetto Python Wheel standard distribuibile su PyPI | ~26 KB |

---

## 🚀 Come Installare

### 🪟 Su Windows

#### Opzione 1: Eseguibile Standalone (Senza bisogno di Python)
Prendi `dist/ngit.exe`, copialo in una cartella a tua scelta inclusa nel `PATH` (ad esempio `C:\Windows\System32` o una cartella `C:\Tools`), e aprilo in qualsiasi terminale digitando:
```cmd
ngit --help
```

#### Opzione 2: Installer Automatico 1-Click (`install.bat`)
Fai doppio click su **`install.bat`**:
- Installa NaturalGit nel tuo ambiente utente.
- Aggiunge automaticamente la cartella al `PATH` di Windows.
- Rende subito disponibile il comando `ngit` in qualsiasi terminale (CMD e PowerShell).

#### Opzione 3: Con Pip / Wheel
```cmd
pip install dist/naturalgit-1.0.0-py3-none-any.whl
```

---

### 🐧 Su Linux (Ubuntu, Debian, Fedora, Arch, CentOS)

#### Opzione 1: Pacchetto Debian / Ubuntu (`.deb`)
```bash
sudo dpkg -i dist/naturalgit_1.0.0_all.deb
# oppure
sudo apt install ./dist/naturalgit_1.0.0_all.deb
```
Il comando `ngit` sarà subito disponibile globalmente in `/usr/bin/ngit`.

#### Opzione 2: Installer Automatico (`install.sh`)
```bash
chmod +x install.sh
./install.sh
```
Installa le dipendenze e configura il collegamento in `~/.local/bin/ngit` e `/usr/local/bin/ngit`.

#### Opzione 3: Con Pip / Wheel
```bash
pip install dist/naturalgit-1.0.0-py3-none-any.whl --break-system-packages
```

---

## 💡 Guida Rapida all'Uso

### 1. Esecuzione Standard (con conferma di sicurezza)
```cmd
ngit "mostra ultimi commit"
```
Il tool identifica il comando, mostra la spiegazione e ti chiede:  
`Eseguire questo comando? [Si/no/modifica]:` (basta premere **Invio** o **S** per eseguire).

### 2. Esecuzione Immediata (`-y` o `--yes`)
Se non vuoi che ti chieda ogni volta conferma, aggiungi `-y`:
```cmd
ngit -y "invia le modifiche al server"
```

### 3. Esecuzione Automatica Permanente (`--auto-confirm`)
Puoi impostare l'esecuzione automatica definitiva senza dover scrivere `-y` ogni volta:
```cmd
ngit config --auto-confirm true
```
Da quel momento qualsiasi comando verrà eseguito immediatamente nella shell (salvo operazioni distruttive come `reset --hard` che chiederanno sempre conferma).

### 4. Simulazione Senza Esecuzione (`-d` o `--dry-run`)
Se vuoi solo vedere quale comando verrebbe eseguito **senza toccare nulla**:
```cmd
ngit -d "elimina forzatamente il branch dev"
```

### 5. Utilizzo del Motore Locale C/Assembly (`-l` o `--local`)
Se vuoi eseguire l'inferenza **100% offline** tramite la libreria nativa `llmasm` (.asmbin / C AVX2 engine):
```cmd
ngit -l "mostra i branch remoti"
```
Se il flag `-l` / `--local` non viene passato, NaturalGit utilizza come sempre il provider e modello configurati (Gemini Flash, Ollama, OpenAI o euristiche).

Per configurare il modello locale:
```cmd
ngit config --local-model C:\percorso\modello.asmbin
ngit config --local-backend c  # oppure "auto" o "numpy"
```

---

## 🤖 Addestramento e Memoria Personalizzata

NaturalGit non si limita a frasi statiche: **impara dalle tue istruzioni**!

### Insegnare una nuova regola (`ngit learn`)
```cmd
ngit learn "metti tutto al sicuro" "git add -A && git commit -m 'Salvataggio rapido'" "Esegue add e commit di sicurezza"
```
Da quel momento risponderà in **meno di 2 millisecondi** e passerà questa regola anche a Gemini come esempio di stile!

### Apprendimento interattivo durante l'uso
Se Gemini propone un comando e premi **`m`** per modificarlo manualmente:  
Dopo l'esecuzione, NaturalGit ti chiederà:  
`💡 Vuoi memorizzare questo comando per la frase '...'? [S/n]`  
Se confermi, viene memorizzato istantaneamente per le prossime volte!

### Gestione delle Regole e della Cache
- **Visualizza le regole apprese:** `ngit rules`
- **Elimina una regola:** `ngit rules --delete "frase"`
- **Svuota la cache delle risposte AI:** `ngit cache --clear`

---

## 📖 Esempi Pratici di Utilizzo

### 1. Ispezione e Differenze
```bash
ngit "cosa ho modificato"
ngit "mostrami i nomi dei file modificati"
ngit "quali file non sono tracciati"
ngit "differenze rispetto all'ultimo commit"
ngit "differenze in staging"
ngit "chi ha modificato il file app.py"
```

### 2. Aggiunta e Commit
```bash
ngit "aggiungi tutto"
ngit "metti in staging il file main.py"
ngit "togli il file main.py dallo staging"
ngit "committa dicendo 'fix problema di login'"
ngit "committa tutto con messaggio 'primo rilascio'"
ngit "modifica l'ultimo commit"
```

### 3. 🌟 Commit Descrittivo Intelligente (con info utente e metriche file)
```bash
# Crea un commit dettagliato con metadati autore, host, branch e metriche (+/- righe per ogni file):
ngit "descrivi un commit con tutte le informazioni dei file e dell'utente"

# Oppure tramite flag rapido con titolo opzionale:
ngit -dc "Nuova release 1.0"
```

### 4. Gestione Branch
```bash
ngit "crea il branch feature/login"
ngit "crea e spostati sul branch dev"
ngit "spostati su main"
ngit "elenca tutti i branch"
ngit "rinomina il branch in bugfix-login"
ngit "elimina il branch feature/vecchio"
ngit "elimina forzatamente il branch bugfix"
```

### 5. Sincronizzazione con il Server Remoto (Push / Pull)
```bash
ngit "invia le modifiche al server"
ngit "scarica le novità"
ngit "scarica solo le informazioni senza applicarle"
ngit "mostrami i server remoti"
```

### 6. Stash (Metti da parte)
```bash
ngit "metti da parte le modifiche temporaneamente"
ngit "salva nello stash con messaggio 'lavoro a metà'"
ngit "recupera le modifiche messe da parte"
ngit "applica lo stash senza cancellarlo"
ngit "mostra l'elenco degli stash"
ngit "cancella lo stash"
```

### 7. Cronologia, Ricerca e Statistiche
```bash
ngit "mostrami gli ultimi 5 commit ad albero"
ngit "cerca nei commit la parola fix"
ngit "mostra i dettagli dell'ultimo commit"
ngit "chi ha fatto più commit"
```

### 8. Annullamento e Ripristino (con avvisi di sicurezza ⚠️)
```bash
ngit "annulla le modifiche al file config.py"
ngit "annulla tutte le modifiche non salvate"
ngit "annulla l'ultimo commit ma tieni i file modificati"
ngit "cancella l'ultimo commit"
ngit "crea un commit che annulla l'ultimo"
```

### 9. Frasi Libere Complesse (Gemini Flash AI)
```bash
ngit "fammi un quadro completo della situazione prima che faccia danni"
ngit "elimina tutti i file che non c'entrano niente con git"
ngit "fammi vedere chi è l'ultimo autore che ha fatto un commit"
```

---

## ⚙️ Configurazione (`ngit config`)

Visualizza la configurazione attuale:
```cmd
ngit config --show
```

Opzioni disponibili:
```cmd
# Imposta o aggiorna la chiave Gemini
ngit config --provider gemini --api-key LA_TUA_CHIAVE

# Imposta il modello a bassissima latenza (predefinito)
ngit config --model gemini-flash-lite-latest

# Attiva esecuzione automatica senza prompt
ngit config --auto-confirm true

# Usa Ollama (100% locale, gratuito e offline)
ngit config --provider ollama --model llama3

# Ripristina impostazioni iniziali
ngit config --reset
```

---

## 🛠️ Opzioni della Riga di Comando

| Opzione | Descrizione |
|---|---|
| `<query>` | L'istruzione in linguaggio naturale (es. `ngit "fai il push"`) |
| `-dc`, `--describe-commit` | Crea un commit descrittivo con tutti i metadati dell'utente e dei file |
| `-y`, `--yes` | Esegue subito il comando senza chiedere conferma (salvo comandi distruttivi) |
| `-d`, `--dry-run` | Mostra solo il comando proposto e la spiegazione senza eseguirlo |
| `-i`, `--interactive` | Avvia la shell interattiva |
| `config` | Gestione delle preferenze e delle chiavi API |
| `learn` | Insegna una nuova regola o associazione personalizzata |
| `rules` | Visualizza o elimina le regole personalizzate |
| `cache` | Gestione della cache delle risposte AI (`--clear`) |

---

## 👤 Autore

- **Nome Utente Git / GitHub:** `SalvatoreTatanka056`
- **Email:** `tatanka056@gmail.com`
- **Repository:** [https://github.com/SalvatoreTatanka056/NaturalGit](https://github.com/SalvatoreTatanka056/NaturalGit)
