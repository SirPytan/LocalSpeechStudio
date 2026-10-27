"""Chatterbox-Multilingual-Worker – läuft in der eigenen Umgebung .venv-chatterbox (benötigt transformers 5.2).

Protokoll siehe worker_client.py. Aufruf: chatterbox_worker.py <modellordner>
Ohne Referenz-Audio wird die eingebaute Standardstimme (conds.pt) verwendet.
"""
import os
import sys
import warnings

from worker_client import worker_main


def load():
    import time

    t0 = time.perf_counter()
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    import torch
    from chatterbox.models.tokenizers import tokenizer as cb_tokenizer
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    class _NoChinese:
        """Der Chinesisch-Konverter lädt beim Start Zusatzdaten aus dem Internet – hier nicht benötigt."""

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, text):
            raise RuntimeError("Chinesisch wird in dieser App nicht unterstützt.")

    cb_tokenizer.ChineseCangjieConverter = _NoChinese

    t1 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_local(sys.argv[1], device, t3_model="v3")
    t2 = time.perf_counter()
    if model.conds is not None:  # Aufwärmen, sonst dauert die erste Generierung ~20 s länger
        model.generate("Hallo.", language_id="de")
    print(f"[Zeiten] Import {t1 - t0:.1f}s | Laden {t2 - t1:.1f}s | Aufwärmen {time.perf_counter() - t2:.1f}s",
          file=sys.stderr, flush=True)
    state = {"torch": torch, "model": model, "device": device, "default": model.conds, "cache": {}}
    vram = torch.cuda.memory_allocated() / 1024**3 if device == "cuda" else 0.0
    return state, {"vram_gb": round(vram, 2), "sampling_rate": model.sr, "device": device}


def generate(state, req):
    torch, model = state["torch"], state["model"]
    exaggeration = float(req.get("exaggeration", 0.5))
    prompt = req.get("audio_prompt")
    if prompt:
        key = (prompt, os.path.getmtime(prompt))
        if key not in state["cache"]:  # Stimmmerkmale nur einmal pro Referenzdatei berechnen
            model.prepare_conditionals(prompt, exaggeration=exaggeration)
            state["cache"] = {key: model.conds}
        model.conds = state["cache"][key]
    else:
        if state["default"] is None:
            raise RuntimeError("Keine Standardstimme (conds.pt) vorhanden – bitte ein Referenz-Audio angeben.")
        model.conds = state["default"]
    seed = int(req.get("seed", 0))
    torch.manual_seed(seed)
    if state["device"] == "cuda":
        torch.cuda.manual_seed_all(seed)
    wav = model.generate(
        req["text"], language_id=req.get("language", "de"), exaggeration=exaggeration,
        cfg_weight=float(req.get("cfg_weight", 0.5)), temperature=float(req.get("temperature", 0.8)),
    )
    return wav.cpu().numpy()


if __name__ == "__main__":
    worker_main(load, generate)
