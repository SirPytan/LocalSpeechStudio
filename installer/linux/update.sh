#!/usr/bin/env bash
# TTS & ASR (lokal) – Update abschließen (nach dem Entpacken einer neuen Version in diesen Ordner).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

echo "======================================================================"
echo " TTS & ASR (lokal) – Update abschließen"
echo "======================================================================"
echo " WOZU: Nach dem Entpacken einer neuen Version bringt update.sh die"
echo " vorhandene Installation auf den Stand der neuen Programmdateien. Die"
echo " neuen Dateien allein reichen nicht: geänderte Python-Pakete und die"
echo " Umstellungen zwischen den Versionen werden erst hiermit angewendet."
echo
echo " SO GEHT EIN UPDATE:"
echo "   1. App beenden (Terminal schließen oder \"exit\" eingeben)."
echo "   2. Das neue Archiv in DIESEN Ordner entpacken, Dateien ersetzen:"
echo "      $PWD"
echo "   3. ./update.sh ausführen."
echo
echo " WAS PASSIERT:"
echo "   - Zeigt installierte und neue Versionsnummer und die Änderungen an."
echo "   - Führt die Update-Schritte jeder neueren Version der Reihe nach aus."
echo "   - Gleicht geänderte Python-Umgebungen ab (lädt nur Neues herunter)."
echo "   - Räumt nicht mehr benötigte Dateien auf (spart Speicherplatz)."
echo " Modelle, gespeicherte Stimmen und Einstellungen bleiben erhalten."
echo " Für eine NEUE Installation stattdessen ./install.sh verwenden."
echo "======================================================================"
echo

interactive=1
case " $* " in *" --non-interactive "*) interactive=0 ;; esac

if [ ! -x .venv/bin/python ]; then
    echo " In diesem Ordner ist noch keine Installation (.venv fehlt). Bitte zuerst ./install.sh ausführen."
    exit 1
fi

while pgrep -f "$PWD/(\.venv|tools)/" >/dev/null 2>&1; do
    echo " Die App läuft noch – bitte beenden (Terminal schließen oder \"exit\" eingeben)."
    [ "$interactive" = 1 ] || exit 1
    read -r -p " Danach Enter drücken ... " _ || exit 1
done

if .venv/bin/python setup_env.py --update "$@"; then
    chmod +x start.sh install.sh update.sh 2>/dev/null || true
    echo
    echo "======================================================================"
    echo " Update abgeschlossen. Starten wie gewohnt mit ./start.sh"
    echo "======================================================================"
else
    echo
    echo "Update nicht abgeschlossen – siehe Meldungen oben. Bei Verbindungsfehlern"
    echo "./update.sh einfach erneut ausführen (bereits Geladenes wird wiederverwendet)."
    exit 1
fi
