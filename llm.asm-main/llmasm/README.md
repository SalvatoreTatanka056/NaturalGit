# llmasm ⚡

**Libreria Python e Motore di Inferenza ad Alte Prestazioni in C per modelli `llm.asm` (`.asmbin`)**

`llmasm` è la libreria Python ufficiale basata sul progetto **`llm.asm`** che unisce la massima velocità del codice nativo compilato in **C/AVX2+FMA** con la comodità, modularità ed eleganza di **Python**.

---

## 🌟 Caratteristiche Principali

- **Motore di Inferenza Nativo in C (`src_c/llmasm_engine.c`)**:
  - Implementazione completa in C99 con istruzioni SIMD **AVX2 + FMA + F16C**, OpenMP e fallback scalare portabile.
  - Dequantizzazione al volo **Q8_0** (blocchi da 32 elementi, formato v1 FP32 e v2 FP16-scale).
  - Kernel matematici ultra-ottimizzati: `RMSNorm`, `RoPE` (split-half Qwen3), `SwiGLU`, `Softmax` e Grouped Query Attention (`GQA`).
  - Tokenizer BPE integrato basato su tabella hash a indirizzamento aperto con hashing di Fibonacci (zero overhead).
  - Gestione efficiente della **KV-Cache** contigua per turno singolo e chat multi-turno.
- **Integrazione Python Trasparente (`ctypes`)**:
  - Nessuna dipendenza da librerie complesse o runtime pesanti.
  - Backend selezionabile (`backend="c"`, `backend="numpy"`, o `backend="auto"`).
  - I sorgenti C sono inclusi nella distribuzione; `auto` compila in una cache utente se trova un compilatore e avvisa quando ripiega su NumPy.
  - Supporto per **generazione in streaming**, campionamento greedy o con temperatura e top-k.
- **Fallback Pure Python / NumPy**:
  - Funziona out-of-the-box su qualsiasi macchina anche in assenza di compilatori C.
- **Tool di Conversione ed Esportazione (`converter.py`)**:
  - Converte modelli Hugging Face Safetensors / BF16 (es. `Qwen/Qwen3-0.6B`) nel formato `.asmbin` compatto a 64-byte alignment.
- **CLI completa**:
  - Esegui prompt, avvia chat interattive, converti pesi o ispeziona modelli direttamente dal terminale.

---

## 📁 Struttura del Progetto

```text
llmasm/
├── src_c/                     # Motore di inferenza in C
│   ├── llmasm_engine.h        # Header C e API esportata (DLL / .so)
│   ├── llmasm_engine.c        # Kernel matematici, Q8_0 AVX2, GQA, BPE
│   ├── Makefile               # Script di compilazione universale Make
│   └── CMakeLists.txt         # Configurazione CMake multipiattaforma
├── llmasm/                    # Pacchetto Python
│   ├── __init__.py            # Export di alto livello (load_model, AsmBinModel, ecc.)
│   ├── c_engine.py            # Binding ctypes verso libllmasm (.dll / .so)
│   ├── c_builder.py           # Auto-compilatore C integrato
│   ├── model.py               # Motore di inferenza Python/NumPy (zero-copy mmap)
│   ├── tokenizer.py           # Tokenizer BPE con hashing di Fibonacci
│   ├── converter.py           # Esportatore Safetensors -> .asmbin
│   └── cli.py                 # Interfaccia a riga di comando (CLI)
├── tests/
│   └── test_llmasm.py         # Test unitari automatizzati
├── pyproject.toml             # Configurazione build Python moderna
└── setup.py                   # Setup pip standard
```

---

## 🚀 Installazione

### 1. Installazione pacchetto Python
Entra nella cartella `llmasm` ed esegui:
```bash
pip install -e .
```

### 2. Compilazione del motore C (Consigliato per massime prestazioni)
Puoi compilare la libreria condivisa nativa (`llmasm.dll` su Windows o `libllmasm.so` su Linux) in uno dei seguenti modi:

Con `backend="auto"` la compilazione è tentata automaticamente al primo uso. Su Windows è necessario MSVC Build Tools con **Desktop development with C++**, eseguendo NaturalGit da **x64 Native Tools PowerShell**; è richiesto un processore AVX2. Se il backend C è obbligatorio, usa `backend="c"`: in assenza del compilatore o della libreria riceverai un errore, non un fallback nascosto. Verifica `model.info["engine"]` per sapere quale backend è attivo.

* **Tramite la CLI di llmasm (automatico):**
  ```bash
  llmasm build-c
  ```
* **Tramite Make (Linux/macOS o MinGW):**
  ```bash
  cd src_c
  make
  ```
* **Tramite CMake:**
  ```bash
  cd src_c
  mkdir build && cd build
  cmake ..
  cmake --build . --config Release
  ```

---

## 💻 Utilizzo in Python

### Esempio Base: Generazione di Testo
```python
import llmasm

# Carica il modello (usa backend="c" per il motore C compilato, oppure "auto")
model = llmasm.load_model("models/qwen3-0.6b-q8_0.asmbin", backend="c")

# Genera una risposta
output = model.generate(
    prompt="Explain how a CPU cache works in 3 sentences.",
    max_tokens=64,
    temperature=0.0  # Greedy decoding
)

print(output)
```

### Generazione in Streaming (Token per Token)
```python
import sys
import llmasm

model = llmasm.load_model("models/qwen3-0.6b-q8_0.asmbin", backend="auto")

prompt = "Scrivi una breve poesia sulla velocità del codice assembly."
print("Risposta: ", end="", flush=True)

for chunk in model.generate(prompt, max_tokens=100, stream=True):
    sys.stdout.write(chunk)
    sys.stdout.flush()
print()
```

### Tokenizzazione Diretta
```python
from llmasm import load_model

model = load_model("models/qwen3-0.6b-q8_0.asmbin")

# Codifica BPE
token_ids = model.encode("Ciao mondo!")
print("Token IDs:", token_ids)

# Decodifica BPE
text = model.decode(token_ids)
print("Decodificato:", text)
```

### Conversione di un modello Hugging Face a `.asmbin`
```python
from llmasm import export_to_asmbin

export_to_asmbin(
    model_dir="models/qwen3-0.6b-bf16",
    out_path="models/qwen3-0.6b-q8_0.asmbin",
    scale_dtype="fp16",  # v2 compatto (consigliato)
    max_seq_len=4096
)
```

---

## 🛠️ Utilizzo tramite CLI (`llmasm`)

Dopo aver installato il pacchetto, hai a disposizione il comando `llmasm`:

```bash
# Eseguire un prompt singolo:
llmasm run -m models/qwen3-0.6b.asmbin -p "Spiega il machine learning in breve"

# Eseguire specificando il backend C nativo:
llmasm run -m models/qwen3-0.6b.asmbin -p "Ciao!" --backend c

# Avviare una chat interattiva multi-turno:
llmasm chat -m models/qwen3-0.6b.asmbin

# Ispezionare i metadati di un file .asmbin:
llmasm info -m models/qwen3-0.6b.asmbin

# Convertire pesi da Safetensors a .asmbin:
llmasm convert --model-dir models/qwen3-bf16 --out models/qwen3-q8_0.asmbin
```

---

## 🧪 Esecuzione dei Test

Per eseguire i test unitari automatizzati su tokenizzazione, quantizzazione e inferenza:
```bash
python -m unittest discover -s tests
```

---

## 👤 Autore
- **Salvatore Romano** (<tatanka056@gmail.com>)
