#!/usr/bin/env bash
# TTS & ASR (lokal) – Installation für Linux (x86_64 / aarch64).
# Erneut ausführen = Installation reparieren / Einstellungen ändern.
# Optionen werden an setup_env.py weitergereicht, z. B. ./install.sh --backend cpu --models-dir ~/tts-models
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

echo "======================================================================"
echo " TTS & ASR (lokal) – Installation"
echo "======================================================================"
echo " Installiert uv, Python 3.12 und alle Pakete in diesen Ordner:"
echo "   $PWD"
echo " Modelle werden NICHT jetzt geladen, sondern später per Klick in der App"
echo " (Bereich „Modelle verwalten“). Der Speicherort der Modelle wird gleich abgefragt."
echo "======================================================================"
echo
free_gb="$(df -Pk . 2>/dev/null | awk 'NR==2 {print int($4 / 1048576)}')"
echo " ACHTUNG – SPEICHERPLATZ"
echo " ----------------------------------------------------------------------"
echo " Die Grundinstallation (ohne Modelle) braucht mindestens 16 GB freien"
echo " Speicher auf diesem Laufwerk (für Downloads und Entpacken). Danach belegt"
echo " sie je nach Grafikkarte ca. 6–10 GB; .venv und tools/uv-cache nutzen"
echo " dieselben Dateien (Hardlinks), \"du\" zählt sie nur einmal."
echo " Später kommen die Modelle dazu (je 0,05–8 GB). Chatterbox, Parler-TTS"
echo " und ClearerVoice-Studio brauchen zusätzlich je ca. 0,5 GB für eine"
echo " eigene Python-Umgebung. Die Größen stehen in der App vor jedem Download."
echo
echo " Freier Speicher hier: ${free_gb:-?} GB"
if [ -n "${free_gb:-}" ] && [ "$free_gb" -lt 16 ]; then
    echo
    echo " !!! Zu wenig freier Speicher – bitte Platz schaffen oder den Ordner"
    echo " !!! auf ein anderes Laufwerk verschieben und install.sh dort starten."
fi
echo
echo " Dauer: je nach Internetverbindung ca. 5–20 Minuten."
echo " ----------------------------------------------------------------------"
case " $* " in
    *" --non-interactive "*) ;;
    *)
        read -r -p "Installation starten? [j/N] " answer || answer=""
        case "$answer" in
            [jJyY]*) ;;
            *) echo "Abgebrochen."; exit 0 ;;
        esac
        ;;
esac
echo

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/tools/uv-cache}"
export UV_PYTHON_INSTALL_DIR="$PWD/tools/python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_HTTP_TIMEOUT=300
UV="$PWD/tools/uv/uv"

fail() {
    echo
    echo "Installation fehlgeschlagen: $1"
    echo "Bei Verbindungsfehlern ./install.sh einfach erneut ausführen (bereits Geladenes wird wiederverwendet)."
    exit 1
}

if [ ! -x "$UV" ]; then
    echo "[1/3] Lade uv (Paketmanager) herunter ..."
    case "$(uname -m)" in
        x86_64|amd64) arch=x86_64 ;;
        aarch64|arm64) arch=aarch64 ;;
        *) fail "Prozessor-Architektur $(uname -m) wird nicht unterstützt." ;;
    esac
    url="https://github.com/astral-sh/uv/releases/latest/download/uv-$arch-unknown-linux-gnu.tar.gz"
    tmp="$(mktemp)"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 6 --retry-delay 3 --retry-all-errors -o "$tmp" "$url" \
            || curl -fL --retry 6 --retry-delay 3 -o "$tmp" "$url" || fail "Download von uv fehlgeschlagen."
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=6 --waitretry=3 -O "$tmp" "$url" || fail "Download von uv fehlgeschlagen."
    else
        fail "Weder curl noch wget gefunden – bitte eines davon installieren (z. B. sudo apt install curl)."
    fi
    mkdir -p tools/uv
    tar -xzf "$tmp" -C tools/uv --strip-components=1 || fail "uv konnte nicht entpackt werden."
    rm -f "$tmp"
fi
"$UV" --version

echo
echo "[2/3] Python 3.12 einrichten ..."
if [ ! -x .venv/bin/python ]; then
    "$UV" venv .venv --python 3.12 --seed || fail "Python-Umgebung konnte nicht angelegt werden."
fi

echo
echo "[3/3] Einstellungen abfragen und Pakete installieren (PyTorch ca. 3 GB, dauert einige Minuten) ..."
.venv/bin/python setup_env.py --env main "$@" || fail "Pakete konnten nicht installiert werden."
# Bei einer bestehenden Installation: versionsabhängige Update-Schritte nachholen.
.venv/bin/python setup_env.py --update --non-interactive || fail "Update-Schritte fehlgeschlagen."
chmod +x update.sh 2>/dev/null || true
chmod +x start.sh 2>/dev/null || true

echo
echo "======================================================================"
echo " Installation abgeschlossen."
echo " Starten: ./start.sh – Modelle in der Weboberfläche unter"
echo " „Modelle verwalten“ mit „Download & Install“ herunterladen."
echo "======================================================================"
case " $* " in
    *" --non-interactive "*) exit 0 ;;
esac
read -r -p "Jetzt starten? [j/N] " answer || answer=""
case "$answer" in
    [jJyY]*) exec ./start.sh ;;
esac
