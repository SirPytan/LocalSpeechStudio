"""Parler-TTS-Worker – läuft in der eigenen Umgebung .venv-parler (benötigt transformers 4.46.1).

Protokoll siehe worker_client.py. Aufruf: parler_worker.py <modell> <beschreibungs-tokenizer>
"""
import importlib.util
import os
import sys
import warnings

from worker_client import worker_main


def load():
    model_dir, description_tokenizer_dir = sys.argv[1], sys.argv[2]
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    import torch
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    from parler_tts import ParlerTTSForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # Ohne accelerate initialisiert transformers 4.46 erst alle Gewichte zufällig auf der CPU (~60 s statt ~7 s).
    fast = {"low_cpu_mem_usage": True} if importlib.util.find_spec("accelerate") else {}
    try:
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=dtype, attn_implementation="sdpa", **fast,
        ).to(device)
    except (TypeError, ValueError):
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_dir, torch_dtype=dtype, **fast).to(device)
    model.eval()
    state = {
        "torch": torch, "model": model, "device": device,
        "prompt_tok": AutoTokenizer.from_pretrained(model_dir),
        "desc_tok": AutoTokenizer.from_pretrained(description_tokenizer_dir),
    }
    vram = torch.cuda.memory_allocated() / 1024**3 if device == "cuda" else 0.0
    return state, {"vram_gb": round(vram, 2), "sampling_rate": model.config.sampling_rate, "device": device}


def generate(state, req):
    torch, model, device = state["torch"], state["model"], state["device"]
    seed = int(req.get("seed", 0))
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    desc = state["desc_tok"](req["description"], return_tensors="pt").to(device)
    prompt = state["prompt_tok"](req["text"], return_tensors="pt").to(device)
    with torch.inference_mode():
        audio = model.generate(
            input_ids=desc.input_ids,
            attention_mask=desc.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
            do_sample=True,
            temperature=float(req.get("temperature", 1.0)),
        )
    return audio.to(torch.float32).cpu().numpy()


if __name__ == "__main__":
    worker_main(load, generate)
