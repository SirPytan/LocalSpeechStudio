"""Weitere Spracherkennungs-Modelle mit derselben Schnittstelle wie Audio8ASR.

- WhisperASR: Whisper-Modelle über transformers (primeline/whisper-large-v3-turbo-german,
  nyrahealth/CrisperWhisper – wortgetreu inkl. Füllwörtern wie „äh“).
- QwenASR: Qwen/Qwen3-ASR-1.7B über das Paket `qwen-asr` (mit --no-deps installiert).

language: englischer Sprachname ("German", "English", ...) oder None für automatische Erkennung.
Lange Aufnahmen werden wie bei Audio8ASR an leisen Stellen in Abschnitte von höchstens 28 s geteilt.
"""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

from audio8_asr import SAMPLE_RATE, split_audio


def _to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    return librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE) if sr != SAMPLE_RATE else wav


def _run_chunks(wav, sr, progress, transcribe_chunk) -> str:
    chunks = split_audio(_to_16k(wav, sr))
    texts = []
    for i, chunk in enumerate(chunks):
        if progress is not None:
            progress(i, len(chunks))
        text = transcribe_chunk(chunk)
        if text:
            texts.append(text)
    return " ".join(texts).strip()


@contextlib.contextmanager
def _filter_native_stdout(prefix: str):
    """Fängt Ausgaben nativer Bibliotheken (fd 1) ab und gibt alle Zeilen außer `prefix` weiter.

    Die Rust-Tokenizer-Bibliothek schreibt bei CrisperWhisper eine harmlose, sehr lange
    „OrderedVocab … holes“-Meldung direkt auf stdout; Python-Filter greifen dort nicht.
    """
    try:
        fd = sys.__stdout__.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    sys.stdout.flush()
    saved = os.dup(fd)
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), fd)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, fd)
            os.close(saved)
            tmp.seek(0)
            kept = [ln for ln in tmp.read().decode("utf-8", "replace").splitlines(True)
                    if not ln.startswith(prefix)]
            if kept:
                sys.stdout.write("".join(kept))
                sys.stdout.flush()


class WhisperASR:
    def __init__(self, processor, model, device):
        self.processor = processor
        self.model = model
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.last_language = None

    @classmethod
    def from_pretrained(cls, path: Path | str, device: str = "cuda", dtype=torch.bfloat16) -> "WhisperASR":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        with _filter_native_stdout("The OrderedVocab you are attempting to save"):
            processor = WhisperProcessor.from_pretrained(str(path))
        model = WhisperForConditionalGeneration.from_pretrained(
            str(path), dtype=dtype, attn_implementation="sdpa",
        ).to(device).eval()
        asr = cls(processor, model, device)
        asr._transcribe_chunk(np.zeros(SAMPLE_RATE, dtype=np.float32), "german")  # Aufwärmen
        return asr

    @torch.inference_mode()
    def _transcribe_chunk(self, wav16k: np.ndarray, language: str | None) -> str:
        features = self.processor(wav16k, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        out = self.model.generate(
            features.to(self.device, self.dtype), language=language, task="transcribe",
            max_new_tokens=440, num_beams=1, do_sample=False, return_timestamps=False,
        )
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def transcribe(self, wav: np.ndarray, sr: int, progress=None, language: str | None = None) -> str:
        lang = language.lower() if language else None
        return _run_chunks(wav, sr, progress, lambda c: self._transcribe_chunk(c, lang))


def _import_qwen_asr():
    # qwen_asr importiert beim Laden den Forced Aligner, der `nagisa` (DyNet) braucht. Den Aligner nutzen
    # wir nicht; fehlt nagisa, genügt daher ein leeres Ersatzmodul.
    try:
        import nagisa  # noqa: F401
    except ImportError:
        sys.modules["nagisa"] = types.ModuleType("nagisa")
    from qwen_asr import Qwen3ASRModel

    return Qwen3ASRModel


class QwenASR:
    def __init__(self, model):
        self.model = model
        self.last_language = None

    @classmethod
    def from_pretrained(cls, path: Path | str, device: str = "cuda", dtype=torch.bfloat16) -> "QwenASR":
        Qwen3ASRModel = _import_qwen_asr()
        model = Qwen3ASRModel.from_pretrained(
            str(path), dtype=dtype, device_map=device, attn_implementation="sdpa",
            max_inference_batch_size=4, max_new_tokens=1024,
        )
        asr = cls(model)
        asr.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE, language="German")  # Aufwärmen
        return asr

    def transcribe(self, wav: np.ndarray, sr: int, progress=None, language: str | None = None) -> str:
        languages = []

        def chunk(c):
            result = self.model.transcribe(audio=(c, SAMPLE_RATE), language=language)[0]
            if result.language:
                languages.append(result.language)
            return (result.text or "").strip()

        text = _run_chunks(wav, sr, progress, chunk)
        self.last_language = languages[0] if languages and not language else None
        return text
