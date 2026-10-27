"""Steuert den Parler-TTS-Worker (eigener Prozess in .venv-parler) aus der Hauptanwendung.

Parler-TTS benötigt transformers 4.46.1, Qwen3-TTS eine neuere Version – deshalb läuft Parler
in einer eigenen Python-Umgebung. "Entladen" beendet den Prozess, womit der VRAM vollständig frei wird.
"""
from __future__ import annotations

import numpy as np

from worker_client import BASE_DIR, WorkerClient

# Sprecher, auf die das Modell für Deutsch trainiert wurde (Model Card, nach Häufigkeit).
GERMAN_SPEAKERS = ["Nicole", "Michelle", "Christopher", "Megan"]


class ParlerClient(WorkerClient):
    NAME = "Parler"
    ENV = "parler"
    SCRIPT = BASE_DIR / "parler_worker.py"
    LOG_FILE = BASE_DIR / "parler_worker.log"

    def generate(self, text: str, description: str, seed: int, temperature: float = 1.0) -> np.ndarray:
        return self.request(text=text, description=description, seed=int(seed), temperature=float(temperature))
