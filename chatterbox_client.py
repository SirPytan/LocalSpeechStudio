"""Steuert den Chatterbox-Multilingual-Worker (eigener Prozess in .venv-chatterbox)."""
from __future__ import annotations

import numpy as np

from worker_client import BASE_DIR, WorkerClient


class ChatterboxClient(WorkerClient):
    NAME = "Chatterbox"
    ENV = "chatterbox"
    SCRIPT = BASE_DIR / "chatterbox_worker.py"
    LOG_FILE = BASE_DIR / "chatterbox_worker.log"

    def generate(self, text: str, audio_prompt: str | None, exaggeration: float, cfg_weight: float,
                 temperature: float, seed: int, language: str = "de") -> np.ndarray:
        return self.request(
            text=text, audio_prompt=audio_prompt, exaggeration=float(exaggeration),
            cfg_weight=float(cfg_weight), temperature=float(temperature), seed=int(seed), language=language,
        )
