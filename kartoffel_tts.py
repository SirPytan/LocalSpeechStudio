"""Kartoffel-Orpheus 3B (deutsch) – Inferenz mit transformers + SNAC-Decoder.

Prompt-Format laut Modellkarte (SebastianBodza/Kartoffel_Orpheus-3B_german_*):
  Natural:   "<Sprecher>: <Text>"
  Synthetic: "<Sprecher> - <Emotion>: <Text>"
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

SAMPLE_RATE = 24000

NATURAL_SPEAKERS = [
    "Jakob", "Anton", "Julian", "Jan", "Alexander", "Emil", "Ben", "Elias", "Felix", "Jonas", "Noah",
    "Maximilian", "Sophie", "Marie", "Mia", "Maria", "Sophia", "Lina", "Lea",
]
SYNTHETIC_SPEAKERS = ["Martin", "Luca", "Anne", "Emma"]
EMOTIONS = [
    "Neutral", "Happy", "Sad", "Excited", "Surprised", "Humorous", "Angry", "Calm", "Disgust", "Fear",
    "Proud", "Romantic",
]
OUTBURSTS = ["haha", "ughh", "wow", "wuhuuu", "ohhh"]

_START_OF_HUMAN = 128259
_END_OF_TEXT = 128009
_END_OF_HUMAN = 128260
_START_OF_AUDIO = 128257
_END_OF_AUDIO = 128258
_AUDIO_OFFSET = 128266
_CODEBOOK = 4096

MAX_CHUNK_CHARS = 280
CHUNK_PAUSE_S = 0.2


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Teilt langen Text an Satzgrenzen in Abschnitte (lange Abschnitte destabilisieren das Modell)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…;:])\s+", text.strip()) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            cut = sentence.rfind(",", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 3 else sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


class _ProgressCriteria(StoppingCriteria):
    def __init__(self, prompt_len: int, callback):
        self.prompt_len = prompt_len
        self.callback = callback

    def __call__(self, input_ids, scores, **kwargs):
        self.callback(input_ids.shape[1] - self.prompt_len)
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


class KartoffelTTS:
    def __init__(self, model, tokenizer, snac, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.snac = snac
        self.device = device

    @classmethod
    def from_pretrained(cls, model_path: Path, snac_path: Path, device: str, dtype) -> "KartoffelTTS":
        if not (snac_path / "pytorch_model.bin").exists():
            raise FileNotFoundError(f"SNAC-Decoder nicht gefunden: {snac_path} – bitte download_models.py ausführen.")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=dtype, device_map=device, attn_implementation="sdpa",
        )
        model.eval()
        snac = SNAC.from_pretrained(str(snac_path)).to(device)
        return cls(model, tokenizer, snac, device)

    def _prompt_ids(self, text: str, speaker: str, emotion: str | None) -> torch.Tensor:
        prefix = f"{speaker} - {emotion}" if emotion else speaker
        ids = self.tokenizer(f"{prefix}: {text}", return_tensors="pt").input_ids
        start = torch.tensor([[_START_OF_HUMAN]], dtype=torch.int64)
        end = torch.tensor([[_END_OF_TEXT, _END_OF_HUMAN]], dtype=torch.int64)
        return torch.cat([start, ids, end], dim=1).to(self.device)

    def _decode(self, generated: torch.Tensor) -> np.ndarray:
        row = generated[0]
        starts = (row == _START_OF_AUDIO).nonzero(as_tuple=True)[0]
        if len(starts):
            row = row[starts[-1].item() + 1:]
        row = row[row != _END_OF_AUDIO]
        codes = (row[: (len(row) // 7) * 7] - _AUDIO_OFFSET).view(-1, 7)
        # Frames mit ungültigen Codes (außerhalb ihres Codebook-Bereichs) verwerfen.
        offsets = torch.arange(7, device=codes.device) * _CODEBOOK
        local = codes - offsets
        codes = local[((local >= 0) & (local < _CODEBOOK)).all(dim=1)]
        if len(codes) == 0:
            return np.zeros(0, dtype=np.float32)
        layers = [
            codes[:, 0].unsqueeze(0),
            codes[:, [1, 4]].reshape(1, -1),
            codes[:, [2, 3, 5, 6]].reshape(1, -1),
        ]
        with torch.inference_mode():
            audio = self.snac.decode([layer.long().to(self.device) for layer in layers])
        return audio.squeeze().float().cpu().numpy()

    def generate(self, text: str, speaker: str, emotion: str | None = None, temperature: float = 0.6,
                 top_p: float = 0.95, repetition_penalty: float = 1.1, progress=None) -> tuple[np.ndarray, int]:
        """Erzeugt Sprache; progress(chunk_index, chunk_count, generated_tokens, max_tokens) optional."""
        chunks = split_text(text)
        if not chunks:
            raise ValueError("Text ist leer.")
        pause = np.zeros(int(SAMPLE_RATE * CHUNK_PAUSE_S), dtype=np.float32)
        parts: list[np.ndarray] = []
        for i, chunk in enumerate(chunks):
            input_ids = self._prompt_ids(chunk, speaker, emotion)
            max_new = min(4000, 400 + 12 * len(chunk))
            criteria = None
            if progress is not None:
                criteria = StoppingCriteriaList([
                    _ProgressCriteria(input_ids.shape[1], lambda n, i=i, m=max_new: progress(i, len(chunks), n, m))
                ])
            with torch.inference_mode():
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=_END_OF_AUDIO,
                    pad_token_id=_END_OF_AUDIO,
                    use_cache=True,
                    stopping_criteria=criteria,
                )
            audio = self._decode(generated[:, input_ids.shape[1]:])
            if len(audio):
                if parts:
                    parts.append(pause)
                parts.append(audio)
        if not parts:
            raise RuntimeError("Das Modell hat kein gültiges Audio erzeugt – bitte erneut versuchen.")
        return np.concatenate(parts), SAMPLE_RATE
