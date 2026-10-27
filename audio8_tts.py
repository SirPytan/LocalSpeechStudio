"""Audio8-TTS Preview 0.6B (Edge0/Audio8-TTS-Preview-0.6b) – Text-to-Speech mit optionalem Stimmklonen.

Das Modell bringt seinen Code im Repo mit (trust_remote_code) und läuft in der Hauptumgebung.
Ohne Referenz wählt es eine zufällige Stimme; bei langen Texten dient der erste Abschnitt als
Referenz für alle weiteren, damit die Stimme gleich bleibt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor, StoppingCriteria

from kartoffel_tts import split_text

MAX_CHUNK_CHARS = 250
MAX_NEW_TOKENS = 1024  # ca. 47 s Audio pro Abschnitt
FRAMES_PER_SECOND = 44100 / 2048


class _ProgressCriteria(StoppingCriteria):
    def __init__(self, max_tokens: int, callback):
        self.prompt_len = None
        self.max_tokens = max_tokens
        self.callback = callback

    def __call__(self, input_ids, scores, **kwargs):
        if self.prompt_len is None:  # erster Aufruf nach dem ersten erzeugten Frame
            self.prompt_len = input_ids.shape[-1] - 1
        self.callback(input_ids.shape[-1] - self.prompt_len, self.max_tokens)
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


class Audio8TTS:
    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self.sample_rate = int(model.config.codec_sample_rate)

    @classmethod
    def from_pretrained(cls, path: Path, device: str, dtype) -> "Audio8TTS":
        processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True)
        model = AutoModel.from_pretrained(str(path), trust_remote_code=True, dtype=dtype).eval().to(device)
        model.load_codec()  # sonst wird der 1,3-GB-Codec erst bei der ersten Generierung geladen
        tts = cls(model, processor, device)
        codes, _ = tts._generate_codes("Hallo.", None, None, 0.7, 0.9, 50, 32, None)  # CUDA-Kernel aufwärmen
        if codes.shape[-1]:
            model.decode_audio(codes)
        return tts

    def _generate_codes(self, text, ref_audio, ref_codes, temperature, top_p, top_k, max_new_tokens, callback):
        kwargs = {}
        if ref_codes is not None:
            kwargs = {"reference_codes": [ref_codes[0]], "reference_text": [ref_codes[1]]}
        elif ref_audio is not None:
            (wav, sr), ref_text = ref_audio
            kwargs = {"reference_audio": [(wav, sr)], "reference_text": [ref_text]}
        inputs = self.processor(text=[text], return_tensors="pt", **kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        criteria = [_ProgressCriteria(max_new_tokens, callback)] if callback else None
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=float(temperature), top_p=float(top_p),
            top_k=int(top_k), do_sample=True, return_dict_in_generate=True, stopping_criteria=criteria,
        )
        return out.codes, bool(out.finished[0])

    @torch.inference_mode()
    def generate(self, text: str, ref_audio=None, ref_text: str | None = None, temperature: float = 0.7,
                 top_p: float = 0.9, top_k: int = 50, seed: int | None = None, progress=None):
        """ref_audio: (wav, sr) oder None. progress(chunk, chunks, tokens, max_tokens). Gibt (wav, sr) zurück."""
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        chunks = split_text(text, MAX_CHUNK_CHARS)
        reference = (ref_audio, ref_text.strip()) if ref_audio is not None else None
        ref_codes = None
        parts = []
        for i, chunk in enumerate(chunks):
            callback = (lambda t, m, i=i: progress(i, len(chunks), t, m)) if progress else None
            codes, _ = self._generate_codes(
                chunk, reference, ref_codes, temperature, top_p, top_k, MAX_NEW_TOKENS, callback,
            )
            if codes.shape[-1] == 0:
                continue
            wav, lengths = self.model.decode_audio(codes)
            parts.append(wav[0, : int(lengths[0])].float().cpu().numpy())
            if reference is None and ref_codes is None:
                ref_codes = (codes[0].cpu(), chunk)  # gleiche Stimme für die folgenden Abschnitte
            if i < len(chunks) - 1:
                parts.append(np.zeros(int(self.sample_rate * 0.2), dtype=np.float32))
        if not parts:
            raise RuntimeError("Das Modell hat kein Audio erzeugt – bitte erneut versuchen.")
        return np.concatenate(parts), self.sample_rate
