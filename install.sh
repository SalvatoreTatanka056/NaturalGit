#!/bin/bash
# ==============================================================================
#  🌿 NaturalGit - Installer per Linux (Ubuntu, Debian, Fedora, Arch, CentOS)
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
fi

# 3. Directory corrente del pacchetto
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📦 Installazione del pacchetto NaturalGit..."

# 4. Installazione tramite pip
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    echo "ℹ pip non trovato, installazione con python3 -m pip..."
    PIP_CMD="python3 -m pip"
fi

# Installa in modalità user o virtualenv evitando errori EXTERNALLY-MANAGED di Debian/Ubuntu moderni
if $PIP_CMD install -e . --no-build-isolation --no-deps 2>/dev/null; then
    echo "✔ Pacchetto installato con successo tramite pip!"
elif $PIP_CMD install -e . --no-build-isolation --no-deps --break-system-packages 2>/dev/null; then
    echo "✔ Pacchetto installato con successo (--break-system-packages)!"
else
    echo "⚠ Installazione fallback creando wrapper in /usr/local/bin o ~/.local/bin..."
fi

# 5. Creazione wrapper eseguibile in ~/.local/bin o /usr/local/bin
INSTALL_BIN="$HOME/.local/bin"
mkdir -p "$INSTALL_BIN"

WRAPPER_PATH="$INSTALL_BIN/ngit"
cat << 'EOF' > "$WRAPPER_PATH"
#!/bin/bash
# Wrapper per NaturalGit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$SCRIPT_DIR/../lib/naturalgit:$PYTHONPATH"
exec python3 -m naturalgit.cli "$@"
EOF
chmod +x "$WRAPPER_PATH"

# Se ha permessi sudo, creiamo il symlink globale
if [ -w /usr/local/bin ]; then
    ln -sf "$WRAPPER_PATH" /usr/local/bin/ngit
    ln -sf "$WRAPPER_PATH" /usr/local/bin/naturalgit
    echo "✔ Collegamenti creati in /usr/local/bin/ngit"
fi

# Verifica se ~/.local/bin è nel PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo ""
    echo "ℹ Suggerimento: Aggiungi ~/.local/bin al tuo PATH aggiungendo questa riga a ~/.bashrc o ~/.zshrc:"
    echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

echo ""
echo "============================================================"
echo "🎉 NaturalGit è pronto! Provalo subito digitando:"
echo "   ngit \"mostrami lo stato dei file\""
echo "   ngit --help"
echo "============================================================"
