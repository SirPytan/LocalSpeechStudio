#!/usr/bin/env bash
# TTS & ASR (lokal) starten. Automatische Modell-Updates abschalten: QWEN_TTS_AUTO_UPDATE=0 ./start.sh
cd "$(dirname "$(readlink -f "$0")")" || exit 1
if [ ! -x .venv/bin/python ]; then
    echo "Die App ist noch nicht installiert – bitte zuerst ./install.sh ausführen."
    exit 1
fi
export PYTHONIOENCODING=utf-8
exec .venv/bin/python app.py "$@"
