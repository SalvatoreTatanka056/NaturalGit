#!/bin/bash
# ==============================================================================
#  🌿 NaturalGit - Installer per Linux (Ubuntu, Debian, Fedora, Arch, CentOS)
#  Include supporto completo al motore locale llmasm in C (AVX2/FMA/OpenMP)
# ==============================================================================

set -e

echo "┌────────────────────────────────────────────────────────────┐"
echo "│  🌿 NaturalGit - Installazione su Linux                    │"
echo "└────────────────────────────────────────────────────────────┘"

# 1. Verifica Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Errore: Python 3 non trovato. Installa python3 prima di continuare."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✔ Rilevato Python $PYTHON_VERSION"

# 2. Verifica Git
if ! command -v git &> /dev/null; then
    echo "⚠ Attenzione: Git non risulta installato. Ricordati di installare git."
else
    echo "✔ Rilevato Git: $(git --version)"
fi

# 3. Directory corrente del pacchetto
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 4. Compilazione C Engine llmasm (opzionale per massima accelerazione SIMD)
if [ -d "$SCRIPT_DIR/llm.asm-main/llmasm/src_c" ]; then
    echo "⚙ Compilazione/verifica motore C ad alte prestazioni per llmasm..."
    if command -v gcc &> /dev/null || command -v clang &> /dev/null; then
        (cd "$SCRIPT_DIR/llm.asm-main/llmasm/src_c" && make > /dev/null 2>&1 || true)
        if [ -f "$SCRIPT_DIR/llm.asm-main/llmasm/src_c/libllmasm.so" ]; then
            mkdir -p "$SCRIPT_DIR/llm.asm-main/llmasm/llmasm"
            cp -f "$SCRIPT_DIR/llm.asm-main/llmasm/src_c/libllmasm.so" "$SCRIPT_DIR/llm.asm-main/llmasm/llmasm/" 2>/dev/null || true
            echo "✔ Motore C nativo compilato con successo (libllmasm.so AVX2/FMA/OpenMP)!"
        fi
    else
        echo "ℹ Compilatore gcc/clang non rilevato. Verrà utilizzato il motore fallback ottimizzato NumPy."
    fi
fi

# 5. Installazione pacchetto tramite pip
echo "📦 Installazione dei pacchetti NaturalGit e llmasm..."
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    echo "ℹ pip non trovato, installazione con python3 -m pip..."
    PIP_CMD="python3 -m pip"
fi

INSTALLED=0
if $PIP_CMD install -e . --no-build-isolation --no-deps 2>/dev/null; then
    echo "✔ Pacchetto installato con successo tramite pip!"
    INSTALLED=1
elif $PIP_CMD install -e . --no-build-isolation --no-deps --break-system-packages 2>/dev/null; then
    echo "✔ Pacchetto installato con successo (--break-system-packages)!"
    INSTALLED=1
elif $PIP_CMD install --user -e . --no-build-isolation --no-deps 2>/dev/null; then
    echo "✔ Pacchetto installato con successo nella directory utente (--user)!"
    INSTALLED=1
else
    echo "ℹ Installazione pip diretta non riuscita (sistema gestito). Configurazione wrapper eseguibile..."
fi

# 6. Creazione wrapper eseguibile in ~/.local/bin
INSTALL_BIN="$HOME/.local/bin"
mkdir -p "$INSTALL_BIN"

create_wrapper() {
    local bin_name="$1"
    local module_entry="$2"
    local wrapper_path="$INSTALL_BIN/$bin_name"

    cat << EOF > "$wrapper_path"
#!/bin/bash
# Wrapper automatico per $bin_name
BASE_REPO="$SCRIPT_DIR"
export PYTHONPATH="\$BASE_REPO:\$BASE_REPO/llm.asm-main/llmasm:\$PYTHONPATH"
exec python3 -m $module_entry "\$@"
EOF
    chmod +x "$wrapper_path"

    # Se ha permessi sudo o root su /usr/local/bin, collega globalmente
    if [ -w /usr/local/bin ]; then
        ln -sf "$wrapper_path" "/usr/local/bin/$bin_name"
    fi
}

create_wrapper "ngit" "naturalgit.cli"
create_wrapper "naturalgit" "naturalgit.cli"
create_wrapper "llmasm" "llmasm.cli"

echo "✔ Launcher creati in $INSTALL_BIN (ngit, naturalgit, llmasm)"
if [ -w /usr/local/bin ]; then
    echo "✔ Collegamenti simbolici globali aggiornati in /usr/local/bin"
fi

# 7. Verifica PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]] && [ ! -w /usr/local/bin ]; then
    echo ""
    echo "ℹ Suggerimento: Aggiungi ~/.local/bin al tuo PATH aggiungendo questa riga a ~/.bashrc o ~/.zshrc:"
    echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

echo ""
echo "============================================================"
echo "🎉 NaturalGit per Linux è pronto all'uso!"
echo "   Prova subito digitando:"
echo "   ngit \"mostra lo stato\""
echo "   ngit help"
echo "   ngit -l \"mostra lo stato\"  # con motore locale llmasm"
echo "============================================================"
