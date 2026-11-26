# 📖 Guida Completa alla Configurazione di NaturalGit (`ngit`)

Questa guida illustra nel dettaglio tutte le opzioni di configurazione, i provider AI supportati (locali e cloud), i parametri prestazionali, i flag da riga di comando e le variabili d'ambiente di **NaturalGit**.

---

## 📑 Indice
1. [Visualizzare la Configurazione Corrente](#1-visualizzare-la-configurazione-corrente)
2. [I Provider Disponibili](#2-i-provider-disponibili)
   - [Provider Locale: `local` / `llmasm` (C + AVX2 / 100% Offline)](#a-provider-locale-local--llmasm-100-offline)
   - [Provider Cloud: `gemini` (Google Gemini Flash)](#b-provider-cloud-gemini)
   - [Provider Locale: `ollama` (Llama 3 / Qwen)](#c-provider-locale-ollama)
   - [Provider: `openai` (OpenAI / LM Studio / LocalAI)](#d-provider-openai-e-compatibili)
   - [Provider Euristico: `heuristic` (< 5ms, senza AI)](#e-provider-euristico-heuristic)
   - [Modalità Automatica: `auto`](#f-modalità-automatica-auto)
3. [Tutti i Parametri di `ngit config`](#3-tutti-i-parametri-di-ngit-config)
4. [Tutti i Flag da Riga di Comando (`ngit`)](#4-tutti-i-flag-da-riga-di-comando-ngit)
5. [Variabili d'Ambiente Supportate](#5-variabili-dambiente-supportate)
6. [Scenari Tipici di Utilizzo ed Esempi Pratici](#6-scenari-tipici-di-utilizzo-ed-esempi-pratici)
7. [Ripristino ai Valori Predefiniti](#7-ripristino-ai-valori-predefiniti)

---

## 1. Visualizzare la Configurazione Corrente

Per visualizzare in qualsiasi momento tutti i parametri attualmente attivi:

```bash
ngit config --show
```
*(Oppure digitando semplicemente `ngit config` senza parametri)*.

Output di esempio:
```text
--- Configurazione NaturalGit ---
  provider: local
  local_model_path: C:\models\qwen3-0.6b-q8_0.asmbin
  local_backend: c
  local_threads: 8
  gemini_api_key: AIza...39Ab
  gemini_model: gemini-flash-lite-latest
  ollama_url: http://localhost:11434
  ollama_model: llama3
  openai_api_key: 
  openai_base_url: https://api.openai.com/v1
  openai_model: gpt-4o-mini
  auto_confirm: False
  language: it
```

---

## 2. I Provider Disponibili

Puoi selezionare il motore di intelligenza artificiale impostando il provider con:
```bash
ngit config --provider <nome_provider>
```

### A. Provider Locale: `local` / `llmasm` (100% Offline)
Usa il motore di inferenza nativo **`llmasm`** scritto in **C con vettorizzazione AVX2 + FMA** (basato su `llm.asm`) che esegue file di modello compatti `.asmbin`. Non richiede connessione Internet, non invia dati all'esterno ed è velocissimo.

```bash
# 1. Imposta il percorso del modello .asmbin (attiva automaticamente il provider 'local'):
ngit config --local-model "C:\percorso\al\tuo\modello.asmbin"

# 2. Scegli il backend (c per C/DLL ad alte prestazioni, numpy per fallback):
ngit config --local-backend c

# 3. Imposta i thread CPU da usare:
ngit config --local-threads 8
```

> [!TIP]
> Una volta configurato `--local-model`, NaturalGit ricorderà questa scelta e userà sempre il modello locale anche se lanci semplicemente `ngit "frase"`. Se invece vuoi usarlo solo per un singolo comando, usa il flag `-l`: `ngit -l "frase"`.

---

### B. Provider Cloud: `gemini`
Usa i modelli Google Gemini tramite API ufficiale. È ottimizzato con prompt a token ridotti (`maxOutputTokens: 120`) e fallback automatico di modello per garantire la minima latenza di risposta (~1.4s).

```bash
# Imposta la chiave API e il provider:
ngit config --provider gemini --api-key "LA_TUA_CHIAVE_GEMINI"

# Modelli disponibili:
ngit config --model gemini-flash-lite-latest   # (Predefinito: il più veloce)
ngit config --model gemini-3.5-flash           # (Massima comprensione su frasi articolate)
ngit config --model gemini-3.1-flash-lite      # (Alternativa ultra-rapida)
```

---

### C. Provider Locale: `ollama`
Usa un server Ollama in esecuzione sul tuo computer o nella tua rete locale. Zero costi, 100% privato.

```bash
# Attiva Ollama:
ngit config --provider ollama

# Modelli consigliati:
ngit config --model llama3               # Bilanciato se hai una GPU
ngit config --model qwen2.5-coder:1.5b    # Leggerissimo e velocissimo anche solo su CPU

# URL personalizzato (se Ollama gira su un'altra macchina):
ngit config --ollama-url http://192.168.1.100:11434
```

---

### D. Provider: `openai` e Compatibili
Supporta le API OpenAI ufficiali o qualsiasi server compatibile OpenAI (come **vLLM**, **LM Studio**, **LocalAI**, **Groq**, **DeepSeek**).

```bash
# Con OpenAI ufficiale:
ngit config --provider openai --api-key "sk-..." --model gpt-4o-mini

# Con LM Studio / vLLM locale:
# Modifica il file ~/.naturalgit/config.json impostando "openai_base_url": "http://localhost:1234/v1"
```

---

### E. Provider Euristico: `heuristic`
Disabilita completamente le chiamate AI e utilizza esclusivamente l'algoritmo euristico locale deterministico (>50 regole in italiano e inglese).
- Tempo di risposta: **< 5 millisecondi**.
- Zero consumi di memoria e CPU.

```bash
ngit config --provider heuristic
```

---

### F. Modalità Automatica: `auto` (Predefinita)
In modalità `auto`, NaturalGit sceglie la strategia migliore in ordine di priorità:
1. Se è configurato un modello locale `local_model_path`, usa il motore locale C.
2. Altrimenti, se è presente una chiave API Gemini, usa Gemini Flash-Lite.
3. Altrimenti, se è configurata una chiave OpenAI, usa OpenAI.
4. Altrimenti, se rileva Ollama attivo, usa Ollama.
5. In assenza di configurazioni o se la rete non è disponibile, scatta il fallback offline con il motore euristico.

```bash
ngit config --provider auto
```

---

## 3. Tutti i Parametri di `ngit config`

| Parametro | Valori ammessi | Descrizione |
| :--- | :--- | :--- |
| `--show` | *(flag)* | Mostra tutti i parametri attivi oscurando le chiavi API. |
| `--provider` | `auto`, `local`, `gemini`, `ollama`, `openai`, `heuristic` | Seleziona il motore di inferenza attivo. |
| `--api-key` | `<string>` | Salva la chiave API per il provider attivo (Gemini o OpenAI). |
| `--model` | `<string>` | Imposta il nome del modello per il provider attualmente attivo. |
| `--local-model` | `<percorso.asmbin>` | Percorso assoluto o relativo del modello binario `.asmbin` per `llmasm`. |
| `--local-backend` | `auto`, `c`, `numpy` | Backend di calcolo: `c` (DLL/SO nativa AVX2), `numpy` (fallback puro). |
| `--local-threads` | `<numero>` | Numero di thread CPU per il calcolo parallelo del motore locale (es. `4`, `8`, `16`). |
| `--gemini-model` | `<nome_modello>` | Imposta specificamente il modello Gemini (es: `gemini-flash-lite-latest`). |
| `--ollama-model` | `<nome_modello>` | Imposta specificamente il modello Ollama (es: `llama3`, `qwen2.5-coder:1.5b`). |
| `--ollama-url` | `<url>` | Indirizzo del server Ollama (default: `http://localhost:11434`). |
| `--openai-model` | `<nome_modello>` | Imposta specificamente il modello OpenAI (es: `gpt-4o-mini`). |
| `--auto-confirm` | `true`, `false` | Se `true`, esegue i comandi senza chiedere conferma `[Si/no/modifica]`. |
| `--reset` | *(flag)* | Ripristina tutti i valori di fabbrica predefiniti. |

---

## 4. Tutti i Flag da Riga di Comando (`ngit`)

Puoi combinare i flag di esecuzione con qualsiasi istruzione:

| Flag | Forma estesa | Descrizione |
| :--- | :--- | :--- |
| `-l` | `--local` | Forza l'uso del motore locale nativo `llmasm` (.asmbin in C) per questo singolo comando. |
| `-j <N>` | `--threads <N>` | Specifica il numero di thread CPU da usare per l'inferenza locale (es. `-j 8`). |
| `-y` | `--yes` | Esegue il comando proposto direttamente **senza chiedere conferma**. |
| `-d` | `--dry-run` | Mostra quale comando verrebbe eseguito e la spiegazione **senza toccare nulla**. |
| `-dc` | `--describe-commit` | Genera un commit descrittivo dettagliato con metadati utente, data, branch e metriche file. |
| `-i` | `--interactive` | Avvia la console REPL interattiva di NaturalGit con cronologia e autocompletamento. |

### Esempi di utilizzo dei flag:
```bash
# Esecuzione immediata senza prompt di conferma:
ngit -y "aggiungi tutti i file e committa dicendo fix login"

# Simulazione sicura:
ngit -d "elimina tutti i file non tracciati"

# Esecuzione locale con 16 thread CPU:
ngit -l -j 16 "fai un merge del branch feature con main"

# Generazione di un commit descrittivo dettagliato:
ngit -dc "implementazione autenticazione a due fattori"
```

---

## 5. Variabili d'Ambiente Supportate

Se preferisci non memorizzare le chiavi nel file JSON su disco (o in ambienti CI/CD come GitHub Actions, GitLab CI, Docker), NaturalGit legge automaticamente:

| Variabile d'Ambiente | Scopo |
| :--- | :--- |
| `GEMINI_API_KEY` | Chiave API di Google Gemini. |
| `OPENAI_API_KEY` | Chiave API di OpenAI. |
| `OPENAI_BASE_URL` | URL base alternativo per OpenAI (es. server locale). |
| `OLLAMA_HOST` | Host e porta del server Ollama. |
| `NGIT_LOCAL_MODEL` | Percorso al modello locale `.asmbin` di `llmasm`. |
| `OMP_NUM_THREADS` | Numero di thread CPU per il motore nativo in C (OpenMP). |

---

## 6. Scenari Tipici di Utilizzo ed Esempi Pratici

### Scenario 1: Sviluppatore 100% Offline e Privato
Vuoi che nessun dato del tuo repository esca dal tuo PC:
```bash
# 1. Imposta il tuo modello locale llmasm
ngit config --local-model "C:\models\qwen3-0.6b-q8_0.asmbin"
ngit config --local-threads 8

# 2. Usa NaturalGit normalmente:
ngit "mostra gli ultimi 5 commit con grafico ad albero"
```

### Scenario 2: Sviluppatore Cloud con Massima Velocità
Vuoi la massima reattività da terminale con il cloud:
```bash
# 1. Configura Gemini
ngit config --provider gemini --api-key "AIzaSy..."

# 2. Attiva l'auto-conferma per la massima produttività:
ngit config --auto-confirm true

# 3. Ora digita qualsiasi istruzione ed eseguila istantaneamente:
ngit "crea il branch feature-auth e spostati lì"
```

### Scenario 3: Server di Integrazione Continua (CI/CD)
Su server remoti dove vuoi solo regole euristiche fisse e deterministiche:
```bash
ngit config --provider heuristic --auto-confirm true
```

---

## 7. Ripristino ai Valori Predefiniti

Se vuoi cancellare tutte le impostazioni e ricominciare da capo:
```bash
ngit config --reset
```

Il file di configurazione fisico si trova in:
- **Windows:** `%USERPROFILE%\.naturalgit\config.json`
- **Linux / macOS:** `~/.naturalgit/config.json`
