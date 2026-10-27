"""ClearerVoice-Studio-Worker – läuft in der eigenen Umgebung .venv-clearvoice (benötigt numpy<2).

Protokoll siehe worker_client.py. Arbeitsverzeichnis ist <Modell-Ordner>/clearvoice, dort liegen
die Checkpoints unter checkpoints/<Modellname> (wie von ClearVoice erwartet).
Die Netze werden erst beim ersten Auftrag für die jeweilige Aufgabe geladen.
"""
import os
import warnings

from worker_client import worker_main

TASKS = {
    "MossFormer2_SE_48K": "speech_enhancement",
    "FRCRN_SE_16K": "speech_enhancement",
    "MossFormerGAN_SE_16K": "speech_enhancement",
    "MossFormer2_SS_16K": "speech_separation",
    "MossFormer2_SR_48K": "speech_super_resolution",
}


def load():
    warnings.filterwarnings("ignore")
    import numpy as np
    import torch
    from clearvoice import ClearVoice

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = {"np": np, "torch": torch, "ClearVoice": ClearVoice, "models": {}, "device": device}
    return state, {"vram_gb": 0.0, "sampling_rate": 48000, "device": device}


def _mono(np, audio):
    audio = np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=0) if audio.ndim > 1 else audio


def generate(state, req):
    np, torch = state["np"], state["torch"]
    name = req["model"]
    if name not in TASKS:
        raise ValueError(f"Unbekanntes ClearVoice-Modell: {name}")
    if not os.path.isfile(os.path.join("checkpoints", name, "last_best_checkpoint")):
        raise FileNotFoundError(f"Checkpoint für {name} fehlt – bitte ClearerVoice-Studio neu installieren.")
    cv = state["models"].get(name)
    if cv is None:
        cv = state["ClearVoice"](task=TASKS[name], model_names=[name])
        state["models"][name] = cv
    with torch.no_grad():
        result = cv(req["input"], online_write=False)
    sr = int(cv.models[0].args.sampling_rate)
    if isinstance(result, list):  # Sprechertrennung: eine Spur je Sprecher
        wav = np.stack([_mono(np, r) for r in result])
    else:
        wav = _mono(np, result)
    if state["device"] == "cuda":
        torch.cuda.empty_cache()
    return wav, sr


if __name__ == "__main__":
    worker_main(load, generate)
