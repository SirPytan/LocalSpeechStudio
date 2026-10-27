"""Steuert den ClearerVoice-Studio-Worker (eigener Prozess in .venv-clearvoice)."""
from __future__ import annotations

import numpy as np

from worker_client import BASE_DIR, MODELS_DIR, WorkerClient

# Anzeigename -> (Modellname, Beschreibung)
CLEARVOICE_MODELS = {
    "Rauschunterdrückung · MossFormer2 48 kHz (empfohlen)": (
        "MossFormer2_SE_48K", "Entfernt Hintergrundgeräusche, volle Bandbreite (48 kHz)."),
    "Rauschunterdrückung · FRCRN 16 kHz": (
        "FRCRN_SE_16K", "Klassisches Entrauschen für Sprache, Ausgabe 16 kHz."),
    "Rauschunterdrückung · MossFormerGAN 16 kHz": (
        "MossFormerGAN_SE_16K", "Starkes Entrauschen, Ausgabe 16 kHz."),
    "Sprecher trennen · MossFormer2 16 kHz": (
        "MossFormer2_SS_16K", "Trennt zwei gleichzeitig sprechende Personen in zwei Spuren (16 kHz)."),
    "Qualität erhöhen · MossFormer2 Super-Resolution 48 kHz": (
        "MossFormer2_SR_48K", "Rechnet dumpfe/niedrig aufgelöste Aufnahmen auf 48 kHz hoch."),
}


class ClearVoiceClient(WorkerClient):
    NAME = "ClearVoice"
    ENV = "clearvoice"
    SCRIPT = BASE_DIR / "clearvoice_worker.py"
    LOG_FILE = BASE_DIR / "clearvoice_worker.log"
    CWD = MODELS_DIR / "clearvoice"

    def process(self, model: str, input_path: str) -> tuple[np.ndarray, int]:
        """Ergebnis: mono-Array bzw. [2, N] bei der Sprechertrennung, dazu die Abtastrate."""
        wav = self.request(model=model, input=str(input_path))
        return wav, self.last_sr
