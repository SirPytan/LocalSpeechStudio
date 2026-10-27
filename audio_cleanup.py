"""Audio-Bereinigung (Rauschunterdrückung) ohne GPU.

- Clear (desert-ant-labs/clear): DeepFilterNet3-Feintuning als ONNX-Modell. Die Merkmalsberechnung
  (STFT, ERB-Bänder, Normalisierung) ist hier in numpy nachgebaut, das Modell läuft über onnxruntime.
- DeepFilterNet (Rikorose/DeepFilterNet): offizielles, vorkompiliertes Kommandozeilenprogramm `deep-filter`.

Beide arbeiten mit 48 kHz; die Eingabe wird automatisch umgerechnet, das Ergebnis ist 48 kHz mono.
"""
from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SR = 48000
FFT_SIZE = 960
HOP = 480
NB_ERB = 32
NB_DF = 96
NORM_ALPHA = 0.99  # libDF calc_norm_alpha(48000, 480, tau=1)
CHUNK_FRAMES = 200  # das Clear-ONNX-Modell erwartet genau 200 Frames (2 s)
CONTEXT_FRAMES = 40  # Überlappung je Blockseite, damit das Modell an Blockgrenzen „eingeschwungen“ ist

CLEAR_VARIANTS = {"studio": "clear-studio.onnx", "natural": "clear-natural.onnx"}

DEEPFILTER_VERSION = "0.5.6"


def _vorbis_window(n: int = FFT_SIZE) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    s = np.sin(0.5 * np.pi * (i + 0.5) / (n // 2))
    return np.sin(0.5 * np.pi * s * s).astype(np.float32)


def _erb_widths(sr: int = SR, fft_size: int = FFT_SIZE, nb_bands: int = NB_ERB, min_nb_freqs: int = 2) -> np.ndarray:
    """Breite (Anzahl FFT-Bins) jedes ERB-Bandes – wie libDF `erb_fb`."""
    def freq2erb(f):
        return 9.265 * np.log1p(f / (24.7 * 9.265))

    def erb2freq(e):
        return 24.7 * 9.265 * (np.exp(e / 9.265) - 1.0)

    freq_width = sr / fft_size
    erb_low, erb_high = freq2erb(0.0), freq2erb(sr // 2)
    step = (erb_high - erb_low) / nb_bands
    widths = np.zeros(nb_bands, dtype=np.int64)
    prev_freq = freq_over = 0
    for i in range(1, nb_bands + 1):
        fb = int(np.floor(erb2freq(erb_low + i * step) / freq_width + 0.5))
        nb_freqs = fb - prev_freq - freq_over
        if nb_freqs < min_nb_freqs:
            freq_over = min_nb_freqs - nb_freqs
            nb_freqs = min_nb_freqs
        else:
            freq_over = 0
        widths[i - 1] = nb_freqs
        prev_freq = fb
    widths[-1] += 1
    too_large = int(widths.sum()) - (fft_size // 2 + 1)
    if too_large > 0:
        widths[-1] -= too_large
    return widths


WINDOW = _vorbis_window()
WNORM = 1.0 / (FFT_SIZE**2 / (2 * HOP))
ERB_WIDTHS = _erb_widths()


def stft(audio: np.ndarray) -> np.ndarray:
    """Analyse wie libDF: 480 Nullen Vorlauf, Vorbis-Fenster, normiert. Ergebnis [T, 481] komplex."""
    n_frames = int(np.ceil(len(audio) / HOP)) + 1  # +1: Signalende vollständig erfassen
    padded = np.zeros(HOP + n_frames * HOP, dtype=np.float32)
    padded[HOP:HOP + len(audio)] = audio
    idx = np.arange(n_frames)[:, None] * HOP + np.arange(FFT_SIZE)[None, :]
    return np.fft.rfft(padded[idx] * WINDOW, axis=-1).astype(np.complex64) * WNORM


def istft(spec: np.ndarray, length: int) -> np.ndarray:
    frames = np.fft.irfft(spec, n=FFT_SIZE, axis=-1).astype(np.float32) * FFT_SIZE * WINDOW
    out = np.zeros(HOP + len(spec) * HOP, dtype=np.float32)
    for t, frame in enumerate(frames):
        out[t * HOP:t * HOP + FFT_SIZE] += frame
    return out[HOP:HOP + length]


def features(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ERB-Merkmale [T, 32] und normierte Spektrum-Merkmale [T, 96] (komplex) – wie DeepFilterNet3."""
    power = spec.real**2 + spec.imag**2
    bounds = np.concatenate([[0], np.cumsum(ERB_WIDTHS)])
    erb = np.stack([power[:, a:b].mean(axis=1) for a, b in zip(bounds[:-1], bounds[1:])], axis=1)
    erb = 10.0 * np.log10(erb + 1e-10)

    erb_feat = np.empty_like(erb, dtype=np.float32)
    state = np.linspace(-60.0, -90.0, NB_ERB, dtype=np.float32)
    for t in range(len(erb)):
        state = erb[t] * (1 - NORM_ALPHA) + state * NORM_ALPHA
        erb_feat[t] = (erb[t] - state) / 40.0

    low = spec[:, :NB_DF]
    spec_feat = np.empty_like(low)
    state = np.linspace(0.001, 0.0001, NB_DF, dtype=np.float32)
    mag = np.abs(low)
    for t in range(len(low)):
        state = mag[t] * (1 - NORM_ALPHA) + state * NORM_ALPHA
        spec_feat[t] = low[t] / np.sqrt(state)
    return erb_feat, spec_feat


def _as_real(x: np.ndarray) -> np.ndarray:
    return np.stack([x.real, x.imag], axis=-1).astype(np.float32)


def _peak_normalize(wav: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    return wav * (10 ** (peak_db / 20) / peak) if peak > 1e-6 else wav


class ClearDenoiser:
    """Clear-Modell (eine Variante). Läuft auf der CPU über onnxruntime."""

    def __init__(self, model_path: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=opts,
                                            providers=["CPUExecutionProvider"])

    def _run(self, spec: np.ndarray, erb_feat: np.ndarray, spec_feat: np.ndarray) -> np.ndarray:
        n = len(spec)
        pad = CHUNK_FRAMES - n
        if pad > 0:
            spec = np.pad(spec, ((0, pad), (0, 0)))
            erb_feat = np.pad(erb_feat, ((0, pad), (0, 0)))
            spec_feat = np.pad(spec_feat, ((0, pad), (0, 0)))
        out = self.session.run(None, {
            "spec": _as_real(spec)[None, None],
            "feat_erb": erb_feat.astype(np.float32)[None, None],
            "feat_spec": _as_real(spec_feat)[None, None],
        })[0][0, 0]
        return (out[..., 0] + 1j * out[..., 1]).astype(np.complex64)[:n]

    def enhance(self, audio: np.ndarray, atten_lim_db: float = 100.0, progress=None) -> np.ndarray:
        """audio: mono float32 mit 48 kHz. atten_lim_db begrenzt die Absenkung (100 = volle Wirkung)."""
        spec = stft(audio)
        erb_feat, spec_feat = features(spec)
        total = len(spec)
        enhanced = np.zeros_like(spec)
        step = CHUNK_FRAMES - 2 * CONTEXT_FRAMES
        starts = [0] if total <= CHUNK_FRAMES else list(range(0, total - CONTEXT_FRAMES * 2, step))
        for i, s in enumerate(starts):
            if progress:
                progress(i, len(starts))
            e = min(s + CHUNK_FRAMES, total)
            out = self._run(spec[s:e], erb_feat[s:e], spec_feat[s:e])
            keep_from = 0 if s == 0 else CONTEXT_FRAMES
            keep_to = len(out) if e == total else len(out) - CONTEXT_FRAMES
            enhanced[s + keep_from:s + keep_to] = out[keep_from:keep_to]
            if e == total:
                break
        if atten_lim_db < 100:
            lim = 10 ** (-abs(atten_lim_db) / 20)
            enhanced = enhanced * (1 - lim) + spec * lim
        return istft(enhanced, len(audio))


# ---------------------------------------------------------------- DeepFilterNet (CLI-Programm)
def deepfilter_asset() -> tuple[str, str]:
    """(Download-URL, Dateiname) des passenden vorkompilierten deep-filter-Programms."""
    machine = platform.machine().lower()
    arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
    if os.name == "nt":
        target = "x86_64-pc-windows-msvc.exe"  # einziger Windows-Build (läuft auf ARM per Emulation)
    elif platform.system() == "Darwin":
        target = f"{arch}-apple-darwin"
    else:
        target = "aarch64-unknown-linux-gnu" if arch == "aarch64" else "x86_64-unknown-linux-musl"
    name = f"deep-filter-{DEEPFILTER_VERSION}-{target}"
    url = f"https://github.com/Rikorose/DeepFilterNet/releases/download/v{DEEPFILTER_VERSION}/{name}"
    return url, "deep-filter.exe" if os.name == "nt" else "deep-filter"


def run_deepfilter(exe: Path, audio: np.ndarray, atten_lim_db: float = 100.0, post_filter: bool = False) -> np.ndarray:
    """audio: mono float32 mit 48 kHz → bereinigtes Audio (gleiche Länge)."""
    import soundfile as sf

    with tempfile.TemporaryDirectory(prefix="dfn_") as tmp:
        src = Path(tmp) / "input.wav"
        sf.write(str(src), audio, SR, subtype="FLOAT")
        cmd = [str(exe), "-D", "-a", f"{float(atten_lim_db):g}", "-o", str(Path(tmp) / "out")]
        if post_filter:
            cmd.append("--pf")
        proc = subprocess.run(cmd + [str(src)], capture_output=True, text=True, errors="replace",
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        result = Path(tmp) / "out" / "input.wav"
        if proc.returncode != 0 or not result.exists():
            raise RuntimeError(f"deep-filter ist fehlgeschlagen (Code {proc.returncode}): "
                               f"{(proc.stderr or proc.stdout).strip()[-400:]}")
        wav, _ = sf.read(str(result), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    return wav[:len(audio)] if len(wav) >= len(audio) else np.pad(wav, (0, len(audio) - len(wav)))


def load_audio_48k(path: str) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(path, sr=SR, mono=True)
    return wav.astype(np.float32)


def finalize(wav: np.ndarray, normalize: bool) -> np.ndarray:
    wav = np.nan_to_num(wav.astype(np.float32))
    if normalize:
        return _peak_normalize(wav)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    return wav / peak * 0.999 if peak > 0.999 else wav
