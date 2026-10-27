"""Spracherkennung mit Audio8-ASR-0.1B bzw. ARK-ASR-0.6B/3B (Edge0/AutoArk, transformers remote code).

Alle drei Größen teilen sich dieselbe Schnittstelle. Das Modell verarbeitet höchstens 30 s am Stück;
längere Aufnahmen werden an leisen Stellen in Abschnitte geteilt und nacheinander transkribiert.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 16000
MAX_CHUNK_SECONDS = 28.0
PROMPT = "Please transcribe this audio."


def split_audio(wav: np.ndarray, sr: int = SAMPLE_RATE, max_seconds: float = MAX_CHUNK_SECONDS) -> list:
    """Teilt lange Aufnahmen bevorzugt an der leisesten Stelle der letzten 8 s jedes Abschnitts."""
    max_len = int(max_seconds * sr)
    chunks, start = [], 0
    while len(wav) - start > max_len:
        search_from = start + max_len - int(8 * sr)
        window = wav[search_from:start + max_len]
        frame = int(0.05 * sr)
        energy = np.convolve(window ** 2, np.ones(frame) / frame, mode="valid")
        cut = search_from + int(np.argmin(energy)) + frame // 2
        chunks.append(wav[start:cut])
        start = cut
    chunks.append(wav[start:])
    return [c for c in chunks if len(c) > int(0.3 * sr)]


class Audio8ASR:
    def __init__(self, processor, tokenizer, model, device):
        self.processor = processor
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self._bad_words_ids = self._build_bad_words_ids(tokenizer)

    @classmethod
    def from_pretrained(cls, path: Path | str, device: str = "cuda", dtype=torch.bfloat16) -> "Audio8ASR":
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        path = str(path)
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                path, trust_remote_code=True, dtype=dtype, attn_implementation="sdpa",
            )
        except (ValueError, NotImplementedError):  # ältere Remote-Code-Varianten kennen nur "eager"
            model = AutoModelForCausalLM.from_pretrained(
                path, trust_remote_code=True, dtype=dtype, attn_implementation="eager",
            )
        model = model.to(device).eval()
        asr = cls(processor, tokenizer, model, device)
        asr._transcribe_chunk(np.zeros(SAMPLE_RATE, dtype=np.float32))  # Aufwärmen
        return asr

    @staticmethod
    def _build_bad_words_ids(tokenizer) -> list:
        eos = tokenizer.eos_token_id
        keep = {eos} if isinstance(eos, int) else set(eos or [])
        bad = set(tokenizer.all_special_ids) - keep
        bad.update(
            tid for tok, tid in tokenizer.get_added_vocab().items()
            if tok.startswith("<") and tok.endswith(">") and tid not in keep
        )
        return [[t] for t in sorted(bad)]

    @torch.inference_mode()
    def _transcribe_chunk(self, wav16k: np.ndarray, max_new_tokens: int = 384) -> str:
        conversation = [{
            "role": "user",
            "content": [{"type": "audio", "array": wav16k}, {"type": "text", "text": PROMPT}],
        }]
        inputs = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, return_tensors="pt", sampling_rate=SAMPLE_RATE,
            audio_padding="longest", text_kwargs={"padding": "longest"}, audio_max_length=30 * SAMPLE_RATE,
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in dict(inputs).items()}
        for key in ("audios", "input_features"):
            if key in inputs and torch.is_floating_point(inputs[key]):
                inputs[key] = inputs[key].to(self.dtype)
        out = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
            bad_words_ids=self._bad_words_ids,
        )
        prompt_len = int(inputs["input_ids"].shape[1])
        return self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()

    def transcribe(self, wav: np.ndarray, sr: int, progress=None, language: str | None = None) -> str:
        """wav: Mono-float32 in beliebiger Abtastrate. progress(i, n) wird vor jedem Abschnitt aufgerufen.

        language wird ignoriert – Audio8-ASR erkennt die Sprache selbst.
        """
        import librosa

        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        chunks = split_audio(wav)
        texts = []
        for i, chunk in enumerate(chunks):
            if progress is not None:
                progress(i, len(chunks))
            text = self._transcribe_chunk(chunk)
            if text:
                texts.append(text)
        return " ".join(texts).strip()
