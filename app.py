# coding=utf-8
"""Lokale Qwen3-TTS Oberfläche (Nachbau der Hugging Face Demo Qwen/Qwen3-TTS).

Modi: Voice Design, Voice Clone (Base), TTS (CustomVoice) – jeweils mit den 1.7B Modellen.
"""
import contextlib
import gc
import html
import json
import os
import sys
import threading
import time
import warnings
from collections import OrderedDict
from pathlib import Path

from version import APP_VERSION, UNVERSIONED

warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")

PORT = int(os.environ.get("PORT", "7860"))
URL = f"http://127.0.0.1:{PORT}/"
_START = time.time()
_STEP_STATE = {"msg": "", "t": _START, "done": False}


def _step(msg: str) -> None:
    if __name__ == "__main__":
        print(f"[{time.time() - _START:5.1f}s] {msg}", flush=True)
        _STEP_STATE.update(msg=msg, t=time.time(), done=msg.startswith("Bereit!"))


def _startup_heartbeat() -> None:
    """Meldet alle 15 s, dass ein langer Startschritt noch läuft (sonst wirkt das Fenster eingefroren)."""
    while not _STEP_STATE["done"]:
        time.sleep(15)
        since = time.time() - _STEP_STATE["t"]
        if not _STEP_STATE["done"] and since >= 14:
            # sys.__stdout__: während des qwen_tts-Imports ist sys.stdout umgeleitet (_silence_import_noise).
            print(f"[{time.time() - _START:5.1f}s]   ... {_STEP_STATE['msg'].rstrip(' .')} – läuft noch "
                  f"({since:.0f}s), bitte warten", file=sys.__stdout__, flush=True)


if __name__ == "__main__":
    import socket

    with socket.socket() as _s:
        if _s.connect_ex(("127.0.0.1", PORT)) == 0:
            print(f"Qwen3-TTS läuft bereits unter {URL} – öffne Browser.")
            if os.environ.get("QWEN_NO_BROWSER") != "1":
                import webbrowser

                webbrowser.open(URL)
            raise SystemExit(0)
    print("=" * 70)
    print(f" TTS & ASR (lokal) {APP_VERSION} wird gestartet – bitte dieses Fenster NICHT schließen.")
    print(" Der erste Start nach der Installation kann mehrere Minuten dauern (Windows prüft")
    print(" dabei die neuen Programmdateien), danach meist unter einer Minute.")
    print("=" * 70, flush=True)
    threading.Thread(target=_startup_heartbeat, daemon=True, name="startup-heartbeat").start()

_step("Lade PyTorch und CUDA ...")
import torch  # noqa: E402
import numpy as np  # noqa: E402

_step("Lade Weboberfläche (Gradio) ...")
import gradio as gr  # noqa: E402
import librosa  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402


@contextlib.contextmanager
def _silence_import_noise():
    """Unterdrückt die flash-attn/SoX-Hinweise beim Import von qwen_tts.

    Bewusst ohne os.dup2 – das zerstört unter Windows die Konsolenausgabe.
    """
    import io
    import logging

    sox_logger = logging.getLogger("sox")
    sox_logger.disabled = True
    real_popen = os.popen

    def quiet_popen(cmd, *args, **kwargs):
        if isinstance(cmd, str) and cmd.strip().lower().startswith("sox"):
            cmd = f"{cmd} 2>{os.devnull}"
        return real_popen(cmd, *args, **kwargs)

    os.popen = quiet_popen
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        os.popen = real_popen
        sox_logger.disabled = False


# Beim Import meldet qwen_tts fehlendes flash-attn und SoX – beides wird nur vom
# (hier nicht genutzten) 25Hz-Tokenizer benötigt.
_step("Lade Qwen-TTS-Bibliothek ...")
with _silence_import_noise():
    from qwen_tts import Qwen3TTSModel
    import kartoffel_tts
    from kartoffel_tts import KartoffelTTS

import app_config  # noqa: E402

INSTALLED_VERSION = app_config.load_config().get("version") or UNVERSIONED
UPDATE_PENDING = INSTALLED_VERSION != APP_VERSION
UPDATE_HINT = (f"Update nicht abgeschlossen: Die Programmdateien sind Version {APP_VERSION}, die Installation ist "
               f"noch auf Version {INSTALLED_VERSION}. Bitte die App beenden und "
               f"{'update.bat' if os.name == 'nt' else './update.sh'} ausführen.")
import audio_cleanup  # noqa: E402
import model_updates  # noqa: E402
import audio8_tts  # noqa: E402
from audio8_tts import Audio8TTS  # noqa: E402
from audio8_asr import Audio8ASR  # noqa: E402
from asr_models import QwenASR, WhisperASR  # noqa: E402
from chatterbox_client import ChatterboxClient  # noqa: E402
from clearvoice_client import CLEARVOICE_MODELS, ClearVoiceClient  # noqa: E402
from parler_client import GERMAN_SPEAKERS as PARLER_GERMAN_SPEAKERS, ParlerClient  # noqa: E402
from worker_client import WorkerClient  # noqa: E402

hf_logging.set_verbosity_error()

BASE_DIR = app_config.BASE_DIR
MODELS_DIR = app_config.MODELS_DIR
LOAD_TIMES_FILE = BASE_DIR / ".load_times.json"
MAX_LOADED_MODELS = int(os.environ.get("QWEN_TTS_MAX_LOADED", "2"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
DEFAULT_LOAD_SECONDS = 30.0

MODEL_LABELS = {
    "VoiceDesign": "Qwen3 Voice Design",
    "Base": "Qwen3 Voice Clone",
    "CustomVoice": "Qwen3 CustomVoice",
    "KartoffelNatural": "Kartoffel Natural",
    "KartoffelSynthetic": "Kartoffel Synthetic",
    "Parler": "Parler-TTS Multilingual",
    "Audio8": "Audio8-TTS 0.6B",
    "Chatterbox": "Chatterbox Multilingual",
    "ASR-WhisperDE": "Whisper Turbo German (0.8B)",
    "ASR-Qwen3": "Qwen3-ASR 1.7B",
    "ASR-Crisper": "Whisper Crisper (1.5B)",
    "ASR-3B": "Audio8-ASR Hoch (3B)",
    "ASR-0.6B": "Audio8-ASR Medium (0.6B)",
    "ASR-0.1B": "Audio8-ASR Low (0.1B)",
    "Clear": "Clear",
    "ClearVoice": "ClearerVoice-Studio",
    "DeepFilterNet": "DeepFilterNet 3",
}
# Sortierung in Listen: alphabetisch, Audio8-ASR aber in der Reihenfolge Hoch, Medium, Low.
SORT_OVERRIDES = {"ASR-3B": "audio8-asr 1", "ASR-0.6B": "audio8-asr 2", "ASR-0.1B": "audio8-asr 3"}


def _sort_key(key: str) -> str:
    return SORT_OVERRIDES.get(key, MODEL_LABELS[key].lower())


MODEL_PATHS = {
    "VoiceDesign": MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Base": MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-Base",
    "CustomVoice": MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "KartoffelNatural": MODELS_DIR / "Kartoffel_Orpheus-3B_german_natural-v0.1",
    "KartoffelSynthetic": MODELS_DIR / "Kartoffel_Orpheus-3B_german_synthetic-v0.1",
    "Parler": MODELS_DIR / "parler-tts-mini-multilingual-v1.1",
    "Audio8": MODELS_DIR / "Audio8-TTS-Preview-0.6b",
    "Chatterbox": MODELS_DIR / "chatterbox",
    "ASR-WhisperDE": MODELS_DIR / "whisper-large-v3-turbo-german",
    "ASR-Qwen3": MODELS_DIR / "Qwen3-ASR-1.7B",
    "ASR-Crisper": MODELS_DIR / "CrisperWhisper",
    "ASR-3B": MODELS_DIR / "ARK-ASR-3B",
    "ASR-0.6B": MODELS_DIR / "ARK-ASR-0.6B",
    "ASR-0.1B": MODELS_DIR / "Audio8-ASR-0.1B",
    "Clear": MODELS_DIR / "clear",
    "ClearVoice": MODELS_DIR / "clearvoice" / "checkpoints",
    "DeepFilterNet": MODELS_DIR / "deepfilternet",
}
ASR_MODELS = {"ASR-WhisperDE", "ASR-Qwen3", "ASR-Crisper", "ASR-3B", "ASR-0.6B", "ASR-0.1B"}
WHISPER_ASR_MODELS = {"ASR-WhisperDE", "ASR-Crisper"}
AUDIO8_ASR_MODELS = {"ASR-3B", "ASR-0.6B", "ASR-0.1B"}
# Laufen auf der CPU ohne Laden (kein VRAM, kein Laden/Entladen).
CPU_TOOLS = {"Clear", "DeepFilterNet"}
# Modelle, die eine eigene Python-Umgebung benötigen (Name in app_config.SIDE_ENVS)
MODEL_ENVS = {"Parler": "parler", "Chatterbox": "chatterbox", "ClearVoice": "clearvoice"}
SIDE_ENV_DOWNLOAD_GB = 3.0  # ungefähr (PyTorch + Pakete); torch wird aus dem gemeinsamen Cache wiederverwendet
# Zusätzlicher Speicherbedarf einer eigenen Python-Umgebung in GB: (real, im Explorer angezeigt). Gleiche Pakete
# (v. a. PyTorch) werden per Hardlink mit der Hauptumgebung bzw. dem uv-Cache geteilt und belegen keinen neuen Platz.
SIDE_ENV_DISK_GB = {
    env: (0.5, 5.5 if os.name == "nt" else 7.0) for env in ("parler", "clearvoice", "chatterbox")
}
DEEPFILTER_EXE = MODELS_DIR / "deepfilternet" / audio_cleanup.deepfilter_asset()[1]
ASR_CHOICES = {MODEL_LABELS[k]: k for k in sorted(ASR_MODELS, key=_sort_key)}
ASR_DEFAULT_CHOICE = MODEL_LABELS["ASR-WhisperDE"]
# (Beschreibung, Lizenz) je Spracherkennungs-Modell.
ASR_DETAILS = {
    "ASR-WhisperDE": ("Whisper large-v3-turbo, auf Deutsch nachtrainiert – schnell und genau. Empfohlen für Deutsch.",
                      "Apache 2.0"),
    "ASR-Qwen3": ("Sehr genau, 30 Sprachen, robust bei Musik und Störgeräuschen.", "Apache 2.0"),
    "ASR-Crisper": ("Schreibt wortgetreu inkl. Füllwörtern wie „äh“ – ideal für Referenztexte beim Stimmklonen.",
                    "CC-BY-NC 4.0 (nur nicht-kommerziell)"),
    "ASR-3B": ("Größtes Audio8-Modell. Übersetzt Deutsch manchmal ins Englische (ignoriert die Sprachauswahl).",
               "Apache 2.0"),
    "ASR-0.6B": ("Schnell und solide.", "Apache 2.0"),
    "ASR-0.1B": ("Sehr klein und schnell, etwas ungenauer.", "CC-BY-NC 4.0 (nur nicht-kommerziell)"),
}
ASR_LANGUAGES = {"Deutsch": "German", "Englisch": "English", "Automatisch erkennen": None}
ASR_DEFAULT_LANGUAGE = "Deutsch"
SNAC_PATH = MODELS_DIR / "snac_24khz"
PARLER_TOKENIZER_PATH = MODELS_DIR / "flan-t5-large"
KARTOFFEL_MODELS = {"KartoffelNatural", "KartoffelSynthetic"}
MODEL_ALIASES = {
    "design": "VoiceDesign", "voicedesign": "VoiceDesign", "vd": "VoiceDesign",
    "clone": "Base", "base": "Base", "vc": "Base",
    "tts": "CustomVoice", "customvoice": "CustomVoice", "cv": "CustomVoice",
    "natural": "KartoffelNatural", "kn": "KartoffelNatural",
    "synthetic": "KartoffelSynthetic", "ks": "KartoffelSynthetic",
    "parler": "Parler", "pt": "Parler",
    "audio8": "Audio8", "a8": "Audio8",
    "chatterbox": "Chatterbox", "cb": "Chatterbox",
    "whisper": "ASR-WhisperDE", "whisper-de": "ASR-WhisperDE", "wde": "ASR-WhisperDE",
    "qwen-asr": "ASR-Qwen3", "qasr": "ASR-Qwen3",
    "crisper": "ASR-Crisper", "crisperwhisper": "ASR-Crisper", "whisper-crisper": "ASR-Crisper",
    "asr-high": "ASR-3B", "asr3b": "ASR-3B",
    "asr-medium": "ASR-0.6B", "asr": "ASR-0.6B", "asr06": "ASR-0.6B",
    "asr-low": "ASR-0.1B", "asr01": "ASR-0.1B",
    "clear": "Clear", "clearvoice": "ClearVoice", "clearervoice": "ClearVoice",
    "deepfilter": "DeepFilterNet", "deepfilternet": "DeepFilterNet", "dfn": "DeepFilterNet",
}
# Projektseite je Modell (sonst erstes passendes Repo im Register = Hauptmodell).
EXTRA_URLS = {
    "ClearVoice": "https://github.com/modelscope/ClearerVoice-Studio",
    "DeepFilterNet": "https://github.com/Rikorose/DeepFilterNet",
}
MODEL_URLS = {
    key: EXTRA_URLS.get(key) or next(r.url for r in model_updates.REPOS if key in r.used_by) for key in MODEL_LABELS
}
MODEL_REPOS = {key: [r for r in model_updates.REPOS if key in r.used_by] for key in MODEL_LABELS}
INSTALLING_MARKER = ".installing"


def _dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3


def _repo_installed(repo) -> bool:
    return repo.local_dir.exists() and not (repo.local_dir / INSTALLING_MARKER).exists()


def _env_installed(env: str) -> bool:
    return app_config.side_env_python(env).exists() and not (
        BASE_DIR / app_config.SIDE_ENVS[env] / INSTALLING_MARKER).exists()


def missing_parts(key: str) -> list:
    """Was für ein Modell noch fehlt: [("env", name) | ("repo", Repo) | ("asset", Pfad)]."""
    parts = []
    if key in MODEL_ENVS and not _env_installed(MODEL_ENVS[key]):
        parts.append(("env", MODEL_ENVS[key]))
    parts += [("repo", r) for r in MODEL_REPOS[key] if not _repo_installed(r)]
    if key == "DeepFilterNet" and not DEEPFILTER_EXE.exists():
        parts.append(("asset", DEEPFILTER_EXE))
    return parts


def is_installed(key: str) -> bool:
    return not missing_parts(key)


def download_size_gb(key: str) -> float:
    total = 0.0
    for kind, item in missing_parts(key):
        total += SIDE_ENV_DOWNLOAD_GB if kind == "env" else item.approx_gb if kind == "repo" else 0.03
    return total


def _fmt_size(gb: float) -> str:
    return f"{gb:.1f} GB" if gb >= 0.1 else f"{gb * 1024:.0f} MB"


def side_env_note(key: str) -> str:
    """Hinweis auf den Zusatzbedarf einer noch fehlenden eigenen Python-Umgebung ('' wenn nicht nötig)."""
    env = MODEL_ENVS.get(key)
    if not env or _env_installed(env):
        return ""
    real, shown = SIDE_ENV_DISK_GB[env]
    shared = (f" (PyTorch wird mit der Hauptumgebung geteilt – der Explorer zeigt den Ordner trotzdem mit "
              f"ca. {shown:.1f} GB an)") if shown > real else ""
    return (f"Braucht eine eigene Python-Umgebung: zusätzlich zum Modell ca. {real:.1f} GB Speicher{shared} "
            "und einige Minuten Einrichtung.")


def _check_disk_space(parts: list) -> None:
    """Bricht ab, wenn auf einem Ziel-Laufwerk nicht genug Platz für die fehlenden Teile ist."""
    import shutil

    needs: dict = {}
    for kind, item in parts:
        target, gb = ((BASE_DIR, SIDE_ENV_DISK_GB[item][0]) if kind == "env"
                      else (MODELS_DIR, item.approx_gb) if kind == "repo" else (MODELS_DIR, 0.03))
        target.mkdir(parents=True, exist_ok=True)
        dev = os.stat(target).st_dev
        path, total = needs.get(dev, (target, 0.0))
        needs[dev] = (path, total + gb)
    for path, gb in needs.values():
        free = shutil.disk_usage(path).free / 1024**3
        if free < gb + 1.0:
            raise RuntimeError(f"Zu wenig Speicherplatz: benötigt ca. {gb + 1.0:.1f} GB, frei sind nur "
                               f"{free:.1f} GB auf dem Laufwerk von {path}.")


MODEL_DISK_GB = {k: _dir_size_gb(p) if is_installed(k) else 0.0 for k, p in MODEL_PATHS.items()}
# Reserve an freiem VRAM, die beim Laden eines Modells übrig bleiben muss (Generierung, Desktop usw.).
# Reicht der Platz nicht, werden zuvor die am längsten ungenutzten Modelle entladen.
VRAM_HEADROOM_GB = float(os.environ.get("QWEN_TTS_VRAM_HEADROOM_GB", "3.0"))


def _free_vram_gb() -> float:
    if DEVICE != "cuda":
        return float("inf")
    free, _ = torch.cuda.mem_get_info()
    if _nvml_handle is not None:  # berücksichtigt auch den Parler-Worker-Prozess
        try:
            free = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle).free
        except Exception:
            pass
    return (free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024**3

SPEAKERS = ["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"]
LANGUAGES = [
    "Auto", "German", "English", "Chinese", "Japanese", "Korean",
    "French", "Spanish", "Portuguese", "Russian", "Italian",
]

_loaded: "OrderedDict[str, Qwen3TTSModel]" = OrderedDict()
_model_sizes_gb: dict = {}
_loading: set = set()
_loading_times: dict = {}  # Modell -> (Startzeit, erwartete Ladedauer in s)
_lock = threading.RLock()


def _read_load_times() -> dict:
    try:
        return json.loads(LOAD_TIMES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_load_time(model_type: str, seconds: float) -> None:
    times = _read_load_times()
    times[model_type] = round(seconds, 1)
    with contextlib.suppress(OSError):
        LOAD_TIMES_FILE.write_text(json.dumps(times), encoding="utf-8")


def _free_cuda() -> None:
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def _unload_locked(model_type: str) -> bool:
    model = _loaded.pop(model_type, None)
    if model is None:
        return False
    if hasattr(model, "close"):
        model.close()
    del model
    _model_sizes_gb.pop(model_type, None)
    _free_cuda()
    print(f"[Modelle] {MODEL_LABELS[model_type]} entladen.", flush=True)
    return True


def unload_model(model_type: str) -> bool:
    with _lock:
        return _unload_locked(model_type)


def unload_all(timeout: float | None = None) -> list:
    """Entlädt alle Modelle. Mit timeout wird nicht unbegrenzt auf laufende Generierungen gewartet."""
    acquired = _lock.acquire(timeout=timeout) if timeout is not None else _lock.acquire()
    if not acquired:
        return []
    try:
        return [m for m in list(_loaded) if _unload_locked(m)]
    finally:
        _lock.release()


def get_model(model_type: str, progress=None):
    """Lädt ein Modell bei Bedarf (mit Fortschrittsanzeige) und entlädt ggf. das am längsten ungenutzte."""
    with _lock:
        if model_type in _loaded:
            _loaded.move_to_end(model_type)
            return _loaded[model_type]

        _wait_for_update(model_type, progress)
        path = MODEL_PATHS[model_type]
        if model_type in CPU_TOOLS:
            raise ValueError(f"{MODEL_LABELS[model_type]} läuft auf der CPU und muss nicht geladen werden.")
        if model_type in _installing:
            raise RuntimeError(f"{MODEL_LABELS[model_type]} wird gerade installiert – bitte warten.")
        if model_type in _uninstalling:
            raise RuntimeError(f"{MODEL_LABELS[model_type]} wird gerade deinstalliert.")
        if not is_installed(model_type):
            raise FileNotFoundError(
                f"{MODEL_LABELS[model_type]} ist nicht installiert – bitte über „⬇ Installieren“ herunterladen."
            )

        needed = MODEL_DISK_GB[model_type] + VRAM_HEADROOM_GB
        while _loaded and (len(_loaded) >= MAX_LOADED_MODELS or _free_vram_gb() < needed):
            _unload_locked(next(iter(_loaded)))

        label = MODEL_LABELS[model_type]
        expected = float(_read_load_times().get(model_type, DEFAULT_LOAD_SECONDS))
        print(f"[Modelle] Lade {label} ...", flush=True)
        result: dict = {}

        def worker():
            try:
                if model_type in KARTOFFEL_MODELS:
                    result["model"] = KartoffelTTS.from_pretrained(path, SNAC_PATH, DEVICE, DTYPE)
                elif model_type == "Parler":
                    result["model"] = ParlerClient.start(path, PARLER_TOKENIZER_PATH)
                elif model_type == "Chatterbox":
                    result["model"] = ChatterboxClient.start(path)
                elif model_type == "ClearVoice":
                    result["model"] = ClearVoiceClient.start()
                elif model_type == "Audio8":
                    result["model"] = Audio8TTS.from_pretrained(path, DEVICE, DTYPE)
                elif model_type in WHISPER_ASR_MODELS:
                    result["model"] = WhisperASR.from_pretrained(path, DEVICE, DTYPE)
                elif model_type == "ASR-Qwen3":
                    result["model"] = QwenASR.from_pretrained(path, DEVICE, DTYPE)
                elif model_type in ASR_MODELS:
                    result["model"] = Audio8ASR.from_pretrained(path, DEVICE, DTYPE)
                else:
                    result["model"] = Qwen3TTSModel.from_pretrained(
                        str(path), device_map=DEVICE, dtype=DTYPE, attn_implementation="sdpa",
                    )
            except BaseException as e:  # noqa: BLE001
                result["error"] = e

        _loading.add(model_type)
        start = time.time()
        _loading_times[model_type] = (start, expected)
        mem_before = torch.cuda.memory_allocated() if DEVICE == "cuda" else 0
        try:
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            while thread.is_alive():
                elapsed = time.time() - start
                if progress is not None:
                    progress(
                        min(0.97, elapsed / expected),
                        desc=f"Lade Modell {label} ... {elapsed:.0f}s (ca. {expected:.0f}s)",
                    )
                thread.join(0.25)
        finally:
            _loading.discard(model_type)
            _loading_times.pop(model_type, None)

        if "error" in result:
            raise result["error"]

        elapsed = time.time() - start
        _save_load_time(model_type, elapsed)
        model = result["model"]
        if isinstance(model, WorkerClient):
            _model_sizes_gb[model_type] = model.vram_gb  # eigener Prozess
        elif DEVICE == "cuda":
            _model_sizes_gb[model_type] = (torch.cuda.memory_allocated() - mem_before) / 1024**3
        else:
            _model_sizes_gb[model_type] = sum(
                p.numel() * p.element_size() for p in model.model.parameters()
            ) / 1024**3
        _loaded[model_type] = model
        _remember_vram(model_type, _model_sizes_gb[model_type])
        print(f"[Modelle] {label} geladen ({elapsed:.1f}s).", flush=True)
        if progress is not None:
            progress(1.0, desc=f"{label} geladen")
        return model


# ---------------------------------------------------------------- Modell-Updates
AUTO_UPDATE = os.environ.get("QWEN_TTS_AUTO_UPDATE", "1") != "0"
_updates: dict = {}  # Modell-Schlüssel -> {"progress": 0..1, "text": str} während ein Update läuft
_update_summary = "Update-Prüfung noch nicht ausgeführt."
_update_running = threading.Lock()


def _wait_for_update(model_type: str, progress=None) -> None:
    """Wartet (mit Fortschritt), bis ein laufendes Update für dieses Modell fertig ist."""
    label = MODEL_LABELS[model_type]
    while model_type in _updates:
        state = _updates.get(model_type) or {}
        if progress is not None:
            progress(state.get("progress", 0.0), desc=f"Update für {label} wird heruntergeladen ... {state.get('text', '')}")
        time.sleep(0.5)


def _labels_for(repo) -> str:
    return ", ".join(MODEL_LABELS[k] for k in repo.used_by)


def _check_all_updates() -> tuple[list, list]:
    """Prüft alle vorhandenen Modelle parallel auf Hugging Face. Gibt (updates, fehler) zurück."""
    from concurrent.futures import ThreadPoolExecutor

    repos = [r for r in model_updates.REPOS if _repo_installed(r)]
    updates, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for repo, fut in [(r, pool.submit(model_updates.check, r)) for r in repos]:
            try:
                info = fut.result()
                if info.needed:
                    updates.append(info)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{repo.repo_id} ({type(e).__name__})")
    return updates, errors


def run_model_updates() -> None:
    """Sucht nach neuen Modellversionen und lädt sie herunter (läuft im Hintergrund-Thread)."""
    global _update_summary
    if not _update_running.acquire(blocking=False):
        print("[Updates] Eine Update-Prüfung läuft bereits.", flush=True)
        return
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
        print("[Updates] Prüfe Modelle auf Hugging Face auf neue Versionen ...", flush=True)
        _update_summary = "Prüfe auf Modell-Updates ..."
        updates, errors = _check_all_updates()
        stamp = time.strftime("%H:%M")
        if errors:
            print(f"[Updates] Prüfung fehlgeschlagen für: {', '.join(errors)} (offline?).", flush=True)
        if not updates:
            _update_summary = (f"Updates zuletzt geprüft um {stamp}: alle Modelle aktuell."
                               if not errors else f"Update-Prüfung um {stamp} unvollständig (keine Verbindung?).")
            if not errors:
                print("[Updates] Alle Modelle sind aktuell.", flush=True)
            return

        # Nur Modelle aktualisieren, die gerade nicht im Speicher sind (Dateien wären sonst gesperrt).
        with _lock:
            todo, deferred = [], []
            for info in updates:
                busy = [k for k in info.repo.used_by if k in _loaded or k in _loading]
                (deferred if busy else todo).append(info)
            for info in todo:
                for key in info.repo.used_by:
                    _updates[key] = {"progress": 0.0, "text": "wartet ..."}
        for info in deferred:
            print(f"[Updates] Neue Version von {info.repo.repo_id} verfügbar, aber {_labels_for(info.repo)} ist "
                  "geladen – Modell entladen und 'update' eingeben (oder beim nächsten Start).", flush=True)

        done, failed = [], []
        for info in todo:
            repo = info.repo
            size = model_updates.format_bytes(info.total_bytes)
            print(f"[Updates] Neue Version: {repo.repo_id} ({len(info.files)} Datei(en), {size}) – "
                  "wird heruntergeladen ...", flush=True)
            last_step = [-1]

            def on_progress(loaded, total, name, repo=repo):
                frac = loaded / total
                text = f"{frac * 100:.0f} % ({model_updates.format_bytes(loaded)} / {model_updates.format_bytes(total)})"
                for key in repo.used_by:
                    _updates[key] = {"progress": frac, "text": text}
                step = int(frac * 10)
                if step != last_step[0]:
                    last_step[0] = step
                    print(f"[Updates] {repo.repo_id}: {text}", flush=True)

            try:
                model_updates.download(info, on_progress)
                done.append(repo.repo_id)
                print(f"[Updates] {repo.repo_id} ist aktuell.", flush=True)
            except Exception as e:  # noqa: BLE001
                failed.append(repo.repo_id)
                print(f"[Updates] Fehler bei {repo.repo_id}: {type(e).__name__}: {e} – alte Version bleibt "
                      "nutzbar, nächster Versuch beim nächsten Start.", flush=True)
            finally:
                still_pending = {k for other in todo[todo.index(info) + 1:] for k in other.repo.used_by}
                for key in repo.used_by:
                    MODEL_DISK_GB[key] = _dir_size_gb(MODEL_PATHS[key])
                    if key not in still_pending:
                        _updates.pop(key, None)

        parts = []
        if done:
            parts.append(f"aktualisiert: {', '.join(done)}")
        if failed:
            parts.append(f"fehlgeschlagen: {', '.join(failed)}")
        if deferred:
            parts.append(f"verschoben (geladen): {', '.join(i.repo.repo_id for i in deferred)}")
        _update_summary = f"Updates um {stamp} – " + "; ".join(parts) + "."
    finally:
        _updates.clear()
        _update_running.release()


def start_model_updates() -> None:
    threading.Thread(target=run_model_updates, daemon=True, name="model-updates").start()


_gpu_name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"


_nvml_handle = None
if DEVICE == "cuda":
    try:
        import pynvml

        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    except Exception:
        _nvml_handle = None


def gpu_stats() -> dict | None:
    """Belegter und gesamter Grafikspeicher der GPU (alle Prozesse) in GB."""
    if DEVICE != "cuda":
        return None
    # NVML sieht (anders als mem_get_info unter Windows/WDDM) auch andere Prozesse wie den Parler-Worker.
    if _nvml_handle is not None:
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
            return {"used": mem.used / 1024**3, "total": mem.total / 1024**3}
        except Exception:
            pass
    free, total = torch.cuda.mem_get_info()
    return {"used": (total - free) / 1024**3, "total": total / 1024**3}


def gpu_summary() -> str:
    stats = gpu_stats()
    if stats is None:
        return "Gerät: CPU"
    return f"GPU-Speicher belegt: {stats['used']:.1f} / {stats['total']:.1f} GB"


def gpu_monitor_html() -> str:
    stats = gpu_stats()
    if stats is None:
        return "<div class='gpu-box'>Keine CUDA-GPU gefunden – läuft auf der CPU.</div>" + tasks_html()
    pct = stats["used"] / stats["total"] * 100 if stats["total"] else 0
    app_gb = torch.cuda.memory_reserved() / 1024**3 + sum(
        m.vram_gb for m in list(_loaded.values()) if isinstance(m, WorkerClient)
    )
    chips = [
        f'<span class="gpu-chip">{html.escape(MODEL_LABELS[k])} <b>{_model_sizes_gb.get(k, 0):.1f} GB</b></span>'
        for k in list(_loaded)
    ] + [f'<span class="gpu-chip gpu-chip-wait">⏳ {html.escape(MODEL_LABELS[k])} lädt ...</span>'
         for k in list(_loading) if k not in _loaded]
    return (
        f"<div class='gpu-box'><div class='gpu-title'>{_gpu_name}</div>"
        '<div class="gpu-row">'
        f'<div class="gpu-label"><span>Grafikspeicher (VRAM)</span>'
        f"<span>{stats['used']:.1f} / {stats['total']:.1f} GB ({pct:.0f} %)</span></div>"
        f'<div class="gpu-track"><div class="gpu-fill" style="width:{min(pct, 100):.1f}%"></div></div>'
        f'<div class="gpu-sub">davon diese App: {app_gb:.1f} GB</div>'
        "</div>"
        f'<div class="gpu-models"><span class="gpu-models-title">Geladene Modelle:</span>'
        f'{"".join(chips) or "<span class=gpu-none>keine</span>"}</div>'
        "</div>" + tasks_html()
    )


_net_sample = {"t": 0.0, "bytes": 0, "rate": 0.0}


def _download_rate() -> float | None:
    """Empfangsrate der Netzwerkkarte in Byte/s (über mind. 1 s gemittelt), None ohne psutil."""
    try:
        import psutil
    except ImportError:
        return None
    now = time.time()
    if now - _net_sample["t"] >= 1.0:
        received = psutil.net_io_counters().bytes_recv
        if _net_sample["t"]:
            _net_sample["rate"] = max(0.0, (received - _net_sample["bytes"]) / (now - _net_sample["t"]))
        _net_sample.update(t=now, bytes=received)
    return _net_sample["rate"]


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d} min"


def _task_html(title: str, frac: float | None, text: str, detail: str = "", started: float | None = None,
               busy: bool = False, expected: float | None = None) -> str:
    esc = html.escape
    right = [] if frac is None else [f"{frac * 100:.0f} %"]
    if started:
        right.append(_fmt_duration(time.time() - started)
                     + (f" / ca. {_fmt_duration(expected)}" if expected else ""))
    width = 100 if frac is None else min(max(frac, 0.0), 1.0) * 100
    fill = "task-fill task-wait" if frac is None else "task-fill task-busy" if busy else "task-fill"
    return (
        '<div class="task">'
        f'<div class="gpu-label"><span class="task-title">{esc(title)}</span><span>{esc(" · ".join(right))}</span></div>'
        f'<div class="gpu-track"><div class="gpu-fill {fill}" style="width:{width:.1f}%"></div></div>'
        f'<div class="gpu-sub">{esc(text)}</div>'
        + (f'<div class="task-detail" title="{esc(detail, quote=True)}">{esc(detail)}</div>' if detail else "")
        + "</div>"
    )


def tasks_html() -> str:
    """Laufende Hintergrund-Aufgaben (Installation, Updates, Laden) – blockiert die Bedienung nicht."""
    rows, downloading = [], False
    for key in list(_install_queue):
        state = _installing.get(key)
        if not state:
            continue
        label = MODEL_LABELS[key]
        if state.get("started") is None:
            rows.append(_task_html(f"⬇️ Installation: {label}", None, "wartet auf die vorherige Installation"))
            continue
        downloading = True
        rows.append(_task_html(f"⬇️ Installation: {label}", state.get("progress", 0.0), state.get("text", ""),
                               state.get("detail", ""), state["started"], state.get("busy", False)))
    updates: dict = {}
    for key, state in list(_updates.items()):
        updates.setdefault((state.get("text", ""), round(state.get("progress", 0.0), 3)), []).append(MODEL_LABELS[key])
    for (text, frac), labels in updates.items():
        waiting = text.startswith("wartet")
        downloading = downloading or not waiting
        rows.append(_task_html(f"🔄 Update: {', '.join(labels)}", None if waiting else frac,
                               "wartet ..." if waiting else text))
    for key, (started, expected) in list(_loading_times.items()):
        elapsed = time.time() - started
        rows.append(_task_html(f"⏳ Laden: {MODEL_LABELS[key]}", min(0.97, elapsed / max(expected, 1.0)),
                               "wird in den Grafikspeicher geladen", started=started, expected=expected))
    for key, started in list(_uninstalling.items()):
        rows.append(_task_html(f"🗑️ Deinstallation: {MODEL_LABELS[key]}", None, "Dateien werden gelöscht ...",
                               started=started))
    if not rows:
        return ""
    rate = _download_rate() if downloading else None
    rate_html = f"<span>Download: {rate / 1024**2:.1f} MB/s</span>" if rate is not None else ""
    return (
        '<div class="gpu-box tasks"><div class="gpu-label"><span class="gpu-title">Hintergrund-Aufgaben</span>'
        f"{rate_html}</div>{''.join(rows)}"
        '<div class="gpu-sub">Läuft im Hintergrund – die App kann währenddessen weiter benutzt werden.</div></div>'
    )


GPU_CSS = """
.gpu-box {font-size: 0.9em; padding: 4px 2px;}
.gpu-title {font-weight: 600; margin-bottom: 4px;}
.gpu-row {margin: 4px 0;}
.gpu-label {display: flex; justify-content: space-between; gap: 8px;}
.gpu-sub {opacity: 0.75; font-size: 0.9em; margin-top: 2px;}
.gpu-track {height: 10px; border-radius: 5px; background: rgba(127,127,127,0.25); overflow: hidden;}
.gpu-fill {height: 100%; border-radius: 5px; background: #3b82f6; transition: width 0.3s ease;}
.gpu-models {display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 8px;}
.gpu-models-title {font-weight: 600; margin-right: 2px;}
.gpu-chip {padding: 2px 8px; border-radius: 999px; background: rgba(59,130,246,0.15);
           border: 1px solid rgba(59,130,246,0.45); white-space: nowrap;}
.gpu-chip-wait {background: rgba(234,179,8,0.15); border-color: rgba(234,179,8,0.5);}
.gpu-none {opacity: 0.7;}
.tasks {margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--border-color-primary);}
.task {margin: 8px 0;}
.task-title {font-weight: 600;}
.task-fill {background: #22c55e;}
.task-busy, .task-wait {background-image: linear-gradient(45deg, rgba(255,255,255,0.28) 25%, transparent 25%,
    transparent 50%, rgba(255,255,255,0.28) 50%, rgba(255,255,255,0.28) 75%, transparent 75%, transparent);
    background-size: 20px 20px; animation: task-stripes 1s linear infinite;}
.task-wait {background-color: rgba(127,127,127,0.45);}
@keyframes task-stripes {from {background-position: 20px 0;} to {background-position: 0 0;}}
.task-detail {font-family: var(--font-mono); font-size: 0.8em; opacity: 0.65; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis;}
"""

# Gleich große Buttons: Modell-Schalter als Raster, Aktionsbuttons neben Auswahllisten mit fester Breite;
# bei schmalem Fenster untereinander in voller Breite.
LAYOUT_CSS = """
.mt {width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 0.92em; margin: 0; border: none !important;}
.mt a {padding: 0 !important; margin: 0 !important;}
.mt tr {border: none !important;}
.mt th.mt-disk {width: 84px;} .mt th.mt-c-vram {width: 74px;} .mt th.mt-c-state {width: 92px;}
.mt th.mt-c-act {width: 140px;}
.mt th, .mt td {padding: 4px 6px; border: none; border-bottom: 1px solid var(--border-color-primary);
                vertical-align: middle; text-align: left; background: none;}
.mt th {font-weight: 600; white-space: nowrap; color: var(--body-text-color-subdued);}
.mt .mt-num {text-align: right; white-space: nowrap;}
.mt .mt-label {font-weight: 600; color: var(--body-text-color);}
.mt a.mt-link {display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px;
    margin: 0 6px 0 0 !important; border-radius: 4px; vertical-align: -4px; text-decoration: none;
    color: var(--body-text-color-subdued);}
.mt a.mt-link:hover {color: var(--color-accent); background: rgba(127,127,127,0.18);}
.mt .mt-info, .mt .mt-muted {color: var(--body-text-color-subdued); font-size: 0.88em;}
.mt .mt-info.mt-warn {color: #d97706;}
.mt .mt-env {color: #d97706; font-size: 0.8em; cursor: help;}
.mt .mt-state {white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.mt .mt-act {text-align: right;}
.mt tr.mt-cat td {font-weight: 700; padding-top: 12px; color: var(--color-accent);
    border-bottom: 2px solid var(--color-accent);}
.mt-btn {width: 96px; height: 26px; padding: 0 6px; border-radius: var(--radius-sm); border: 1px solid transparent;
         font-size: 0.88em; font-weight: 600; cursor: pointer; white-space: nowrap;}
.mt-btn.mt-primary {background: var(--button-primary-background-fill); color: var(--button-primary-text-color);}
.mt-btn.mt-primary:hover {background: var(--button-primary-background-fill-hover);}
.mt-btn.mt-stop {background: var(--button-cancel-background-fill); color: var(--button-cancel-text-color);}
.mt-btn.mt-stop:hover {background: var(--button-cancel-background-fill-hover);}
.mt-btn.mt-secondary {background: var(--button-secondary-background-fill); color: var(--button-secondary-text-color);}
.mt-btn.mt-del {width: 28px; padding: 0; margin-left: 4px; background: none; border-color: var(--border-color-primary);
    color: var(--body-text-color-subdued); font-weight: 400; display: inline-flex; align-items: center; justify-content: center;
    vertical-align: middle;}
.mt-btn.mt-del:hover {color: #fff; background: #dc2626; border-color: #dc2626;}
.mt-btn:disabled {opacity: 0.55; cursor: not-allowed;}
.hidden-toggles {display: none !important;}
.panel-footer {align-items: center !important;}
.panel-footer > .unload-all {flex: 0 0 220px !important; min-width: 220px !important;}
@media (max-width: 600px) {
  .mt {font-size: 0.86em;}
  .mt .mt-disk, .mt .mt-c-state, .mt .mt-state {width: 0 !important; padding: 0 !important; font-size: 0;
      overflow: hidden;}
  .mt th.mt-c-vram {width: 58px;} .mt th.mt-c-act {width: 116px;} .mt-btn {width: 80px;} .mt-btn.mt-del {width: 26px; margin-left: 3px;}
  .mt th, .mt td {padding: 4px 3px;}
}
@media (max-width: 760px) {
  .panel-footer {flex-wrap: wrap !important;}
  .panel-footer > *, .panel-footer > .unload-all {flex: 1 1 100% !important; min-width: 100% !important;}
}
.cat-tabs > .tab-wrapper .tab-container[role=tablist] {display: flex !important; width: 100%;}
.cat-tabs .tab-container.visually-hidden > button {width: 1px !important; min-width: 0 !important;
    padding: 0 !important; overflow: hidden !important;}
@media (max-width: 600px) {
  .cat-tabs .tab-container[role=tablist] {height: auto !important;}
  .cat-tabs .tab-container[role=tablist] > button {white-space: normal !important; min-width: 0 !important;
      flex: 1 1 0 !important; justify-content: center; text-align: center;
      font-size: 0.9em !important; line-height: 1.15; padding: 4px 2px !important; height: auto !important;}
  .asr-row .form {flex-wrap: wrap !important;}
  .asr-row .form > * {flex: 1 1 100% !important; min-width: 100% !important;}
}
.cat-tabs > .tab-wrapper .tab-container[role=tablist] > button {flex: 1 1 0 !important; min-width: 0 !important;
    justify-content: center; font-weight: 600; font-size: 1.05em;}
.voice-row {align-items: stretch !important;}
.voice-row > .voice-action {align-self: stretch !important; height: auto !important; min-height: 46px;
    white-space: nowrap; flex: 0 0 170px !important; min-width: 170px !important; max-width: 170px !important;}
.btn-pair {gap: 8px !important; flex-wrap: nowrap !important;}
.btn-pair > button {flex: 1 1 0 !important; min-width: 0 !important;}
button.gen-btn, .gen-btn {background: #16a34a !important; border-color: #16a34a !important; color: #fff !important;
    font-weight: 700;}
button.gen-btn:hover, .gen-btn:hover {background: #15803d !important; border-color: #15803d !important;}
.status-line textarea, .status-line input {font-size: 0.9em !important;}
.hint, .hint p, .hint li {color: var(--body-text-color-subdued) !important; font-size: 0.9em !important;}
.hint p {margin: 0 !important;}
.section-title h4 {margin: 0 !important;}
.app-header p {color: var(--body-text-color-subdued);}
.tab-header {flex-wrap: wrap !important; align-items: flex-start !important; margin-bottom: 4px;}
.tab-header > :first-child {flex: 1 1 320px !important; min-width: 0 !important; gap: 4px !important;}
.tab-header h3 {margin: 0 !important;}
.tab-header > .hf-link {flex: 0 0 220px !important; min-width: 220px !important; max-width: 220px !important;
    height: 36px !important; align-self: flex-start !important; white-space: nowrap;}
@media (max-width: 760px) {
  .tab-header > :first-child {flex-basis: 100% !important;}
  .tab-header > .hf-link {flex: 1 1 0 !important; min-width: 0 !important; max-width: none !important;}
  .voice-row {flex-wrap: wrap !important;}
  .voice-row > :not(.voice-action) {flex: 1 1 100% !important; min-width: 100% !important;}
  .voice-row > .voice-action {flex: 1 1 0 !important; min-width: 0 !important; max-width: none !important;
      height: 40px !important;}
}
.model-area {position: relative; gap: var(--layout-gap);}
.install-overlay {position: absolute !important; inset: 0; z-index: 50; align-items: center;
    padding: 48px 12px 12px; border-radius: var(--radius-lg);
    background: color-mix(in srgb, var(--body-background-fill) 78%, transparent); backdrop-filter: blur(3px);}
.install-card {max-width: 520px; width: 100%; flex-grow: 0 !important; padding: 18px 20px; gap: 12px;
    border: 1px solid var(--border-color-primary); border-radius: var(--radius-lg);
    background: var(--block-background-fill); box-shadow: 0 6px 24px rgba(0,0,0,0.35); text-align: center;}
.install-card h3 {margin: 0 0 6px !important;}
.mt .mt-ready {white-space: nowrap;}
"""

# Die Tabelle ist reines HTML; ihre Buttons lösen die versteckten Gradio-Buttons (elem_id toggle-<key>) aus.
TOGGLE_JS = """
(() => {
  document.addEventListener('click', (e) => {
    const b = e.target.closest('button[data-toggle], button[data-install], button[data-uninstall]');
    if (!b || b.disabled) return;
    if (b.dataset.uninstall && !window.confirm(b.dataset.confirm)) return;
    const id = b.dataset.toggle ? 'toggle-' + b.dataset.toggle
      : b.dataset.install ? 'install-' + b.dataset.install : 'uninstall-' + b.dataset.uninstall;
    const t = document.getElementById(id);
    const btn = t && (t.tagName === 'BUTTON' ? t : t.querySelector('button'));
    if (btn) { b.disabled = true; btn.click(); }
  });
})();
"""


MODEL_CATEGORIES = [
    ("🎨 Voice Designer", ("VoiceDesign", "Parler")),
    ("🎙️ Voice Cloner", ("Base", "Audio8", "Chatterbox")),
    ("🗣️ Voice Generator", ("CustomVoice", "KartoffelNatural", "KartoffelSynthetic")),
    ("📝 Speech-to-Text", ("ASR-WhisperDE", "ASR-Qwen3", "ASR-Crisper", "ASR-3B", "ASR-0.6B", "ASR-0.1B")),
    ("🧹 Audio-Cleanup", ("Clear", "ClearVoice", "DeepFilterNet")),
]
MODEL_INFO = {
    "VoiceDesign": "Stimme per Beschreibung erzeugen (Qwen3, 10 Sprachen)",
    "Parler": "Stimme per Beschreibung, mehrsprachig inkl. Deutsch",
    "Base": "Klon aus Referenz-Audio + Transkript",
    "Audio8": "Platz 1 der TTS Arena – Klon oder zufällige Stimme",
    "Chatterbox": "Klon ohne Transkript, regelbare Ausdrucksstärke",
    "CustomVoice": "9 vordefinierte Sprecher mit Stilanweisung",
    "KartoffelNatural": "Deutsche Sprecher (natürlich), mit Emotionen",
    "KartoffelSynthetic": "Deutsche Sprecher (synthetisch), mit Emotionen",
    "ASR-WhisperDE": "auf Deutsch optimiert, schnell",
    "ASR-Qwen3": "sehr genau, 30 Sprachen",
    "ASR-Crisper": "wortgetreu inkl. „äh“ · nicht-kommerziell",
    "ASR-3B": "übersetzt Deutsch teils ins Englische",
    "ASR-0.6B": "schnell und solide",
    "ASR-0.1B": "sehr klein · nicht-kommerziell",
    "Clear": "Rauschen entfernen, sehr schnell (CPU)",
    "ClearVoice": "Entrauschen, Sprecher trennen, Qualität erhöhen",
    "DeepFilterNet": "starke Rauschunterdrückung (CPU)",
}
# Geschätzter VRAM-Bedarf (GB); wird nach jedem Laden durch den gemessenen Wert ersetzt.
MODEL_VRAM_FILE = BASE_DIR / ".model_vram.json"
MODEL_VRAM_DEFAULT_GB = {
    "VoiceDesign": 3.9, "Base": 3.9, "CustomVoice": 3.9, "KartoffelNatural": 6.2, "KartoffelSynthetic": 6.2,
    "Parler": 1.8, "Audio8": 2.0, "Chatterbox": 3.0, "ASR-WhisperDE": 1.5, "ASR-Qwen3": 3.8, "ASR-Crisper": 2.9,
    "ASR-3B": 7.7, "ASR-0.6B": 2.3, "ASR-0.1B": 0.7, "ClearVoice": 1.0,
}


def _read_vram_estimates() -> dict:
    try:
        return {**MODEL_VRAM_DEFAULT_GB, **json.loads(MODEL_VRAM_FILE.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return dict(MODEL_VRAM_DEFAULT_GB)


_vram_estimates = _read_vram_estimates()


def _remember_vram(model_type: str, gb: float) -> None:
    if gb <= 0.05:
        return
    _vram_estimates[model_type] = round(gb, 2)
    with contextlib.suppress(OSError):
        MODEL_VRAM_FILE.write_text(json.dumps(_vram_estimates), encoding="utf-8")


def _model_state(key: str) -> str:
    if key in _uninstalling:
        return "wird deinstalliert ..."
    if key in _installing:
        return f"wird installiert – {_installing.get(key, {}).get('text', '')}"
    if key in _install_errors:
        return f"Installation fehlgeschlagen: {_install_errors[key]}"
    if not is_installed(key):
        extra = (f" + eigene Python-Umgebung ca. {SIDE_ENV_DISK_GB[MODEL_ENVS[key]][0]:.1f} GB"
                 if side_env_note(key) else "")
        return f"nicht installiert (Download ca. {_fmt_size(download_size_gb(key))}{extra})"
    if key in CPU_TOOLS:
        return "bereit (läuft auf der CPU, kein Laden nötig)"
    if key in _updates:
        return f"Update wird geladen – {_updates.get(key, {}).get('text', '')}"
    if key in _loading:
        return "wird geladen ..."
    if key in _loaded:
        return f"geladen – belegt {_model_sizes_gb.get(key, 0):.1f} GB VRAM"
    return "nicht geladen"


LINK_ICON_SVG = (
    '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 3h7v7"/><path d="M10 14 21 3"/>'
    '<path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5"/></svg>'
)


TRASH_SVG = ('<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" '
             'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
             '<path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v6M14 11v6"/></svg>')


def _model_row_html(key: str) -> str:
    esc = html.escape
    disk = MODEL_DISK_GB[key]
    installed = is_installed(key) and key not in _installing
    disk_html = _fmt_size(disk) if installed else f'<span class="mt-muted">≈ {_fmt_size(download_size_gb(key))}</span>'
    if key in CPU_TOOLS:
        vram = '<span class="mt-muted">CPU</span>'
    elif key in _loaded:
        vram = f"<b>{_model_sizes_gb.get(key, 0):.1f} GB</b>"
    else:
        vram = f'<span class="mt-muted">≈ {_vram_estimates.get(key, 0):.1f} GB</span>'
    action = "toggle"
    if key in _uninstalling:
        icon, state, btn = "🗑️", "entfernt ...", ("⏳ Entfernt", "secondary", False)
    elif key in _installing:
        pct = _installing[key].get("progress", 0.0) * 100
        icon, state, btn = "⏳", f"{pct:.0f} %", ("⏳ Installiert", "secondary", False)
    elif not installed:
        failed = key in _install_errors
        icon, state = ("❌", "Fehler") if failed else ("⬇️", "fehlt")
        btn, action = ("⬇ Installieren", "primary", True), "install"
    elif key in CPU_TOOLS:
        icon, state, btn = "✅", "bereit", None
    elif key in _updates:
        icon, state, btn = "⬇️", "Update", ("⬇ Update", "secondary", False)
    elif key in _loading:
        icon, state, btn = "⏳", "lädt ...", ("⏳ Lädt", "secondary", False)
    elif key in _loaded:
        icon, state, btn = "✅", "geladen", ("⏏ Entladen", "stop", True)
    else:
        icon, state, btn = "⬜", "aus", ("▶ Laden", "primary", True)
    title = esc(_model_state(key), quote=True)
    if btn is None:
        button = '<span class="mt-muted mt-ready">sofort bereit</span>'
    else:
        text, variant, enabled = btn
        button = (f'<button class="mt-btn mt-{variant}" data-{action}="{esc(key, quote=True)}"'
                  f'{"" if enabled else " disabled"}>{text}</button>')
    if key not in _uninstalling and key not in _installing and key not in _updates and key not in _loading \
            and _uninstall_targets(key):
        confirm = (f"{MODEL_LABELS[key]} deinstallieren?\n\nDie Modelldateien"
                   + (" und die eigene Python-Umgebung" if MODEL_ENVS.get(key) else "")
                   + " werden gelöscht. Später lässt sich das Modell jederzeit wieder installieren.")
        button += (f'<button class="mt-btn mt-del" data-uninstall="{esc(key, quote=True)}" '
                   f'data-confirm="{esc(confirm, quote=True)}" title="Deinstallieren (Speicher freigeben)" '
                   f'aria-label="Deinstallieren">{TRASH_SVG}</button>')
    env_note = side_env_note(key) if not installed else ""
    if env_note:
        disk_html += f'<div class="mt-env" title="{esc(env_note, quote=True)}">+ Umgebung</div>'
    return (
        f'<tr><td class="mt-name"><a class="mt-link" href="{esc(MODEL_URLS[key], quote=True)}" target="_blank" '
        f'rel="noopener" title="Modellseite öffnen" aria-label="Modellseite öffnen">{LINK_ICON_SVG}</a>'
        f'<span class="mt-label">{esc(MODEL_LABELS[key])}</span>'
        f'<div class="mt-info">{esc(MODEL_INFO.get(key, ""))}</div>'
        + (f'<div class="mt-info mt-warn">⚠️ {esc(env_note)}</div>' if env_note else "")
        + "</td>"
        f'<td class="mt-num mt-disk">{disk_html}</td>'
        f'<td class="mt-num">{vram}</td>'
        f'<td class="mt-state" title="{title}">{icon}<span class="mt-state-text"> {state}</span></td>'
        f'<td class="mt-act">{button}</td></tr>'
    )


def model_table_html() -> str:
    body = []
    for title, keys in MODEL_CATEGORIES:
        body.append(f'<tr class="mt-cat"><td colspan="5">{html.escape(title)}</td></tr>')
        body.extend(_model_row_html(k) for k in sorted(keys, key=_sort_key))
    return (
        '<table class="mt"><thead><tr><th>Modell</th><th class="mt-num mt-disk">Größe</th>'
        '<th class="mt-num mt-c-vram">VRAM</th><th class="mt-c-state">Status</th><th class="mt-c-act"></th></tr>'
        "</thead><tbody>"
        + "".join(body) + "</tbody></table>"
    )


def status_note_markdown() -> str:
    errors = "".join(f"  \n❌ **{MODEL_LABELS[k]}:** {v}" for k, v in list(_install_errors.items()))
    return (
        f"Größe: Platz auf der Festplatte (≈ = Download, solange das Modell fehlt). VRAM: Messwert bei geladenen "
        f"Modellen, sonst Schätzung (≈). Max. {MAX_LOADED_MODELS} Modelle gleichzeitig; reicht der freie VRAM nicht "
        f"(mind. {VRAM_HEADROOM_GB:.1f} GB Reserve), wird vorher das am längsten ungenutzte entladen.  \n"
        f"📁 Modell-Ordner: `{MODELS_DIR}`  \n🔄 {_update_summary}{errors}"
    )


SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".aif", ".aiff", ".aifc", ".caf", ".w64", ".rf64"}
FORMATS_HINT = (
    "**Unterstützte Formate:** WAV, MP3, FLAC, OGG (Vorbis/Opus), AIFF, CAF · "
    "**nicht unterstützt:** M4A/AAC, WMA, MP4/Video (vorher z. B. in WAV/MP3 umwandeln). "
    "Empfehlung: 5–15 s saubere Sprache ohne Hintergrundgeräusche, Mono oder Stereo, beliebige Abtastrate."
)


def _load_ref_audio(path: str):
    """Referenz-Audio als Mono-float32 laden (über libsndfile, kein ffmpeg nötig)."""
    ext = Path(path).suffix.lower()
    if ext and ext not in SUPPORTED_AUDIO_EXTS:
        raise ValueError(
            f"Format '{ext}' wird nicht unterstützt. Erlaubt: WAV, MP3, FLAC, OGG/Opus, AIFF, CAF."
        )
    try:
        wav, sr = librosa.load(path, sr=None, mono=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Audiodatei konnte nicht gelesen werden ({type(e).__name__}). "
                         "Bitte als WAV, MP3 oder FLAC speichern.") from e
    if len(wav) < sr * 1:
        raise ValueError("Referenz-Audio ist kürzer als 1 Sekunde.")
    return wav.astype(np.float32), int(sr)


def _to_int16(wav) -> np.ndarray:
    return (np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0) * 32767).astype(np.int16)


def _log_request(mode: str, **params) -> None:
    details = " | ".join(f"{k}={v!r}" for k, v in params.items())
    print(f"[{mode}] {details}", flush=True)


def generate_voice_design(text, language, voice_description, progress=gr.Progress()):
    if not text or not text.strip():
        return None, "Fehler: Text ist erforderlich.", None
    if not voice_description or not voice_description.strip():
        return None, "Fehler: Stimmbeschreibung ist erforderlich.", None
    try:
        _log_request("Voice Design", sprache=language, beschreibung=voice_description.strip(), text=text.strip()[:80])
        with _lock:
            model = get_model("VoiceDesign", progress)
            progress(None, desc="Generiere Audio ...")
            start = time.time()
            wavs, sr = model.generate_voice_design(
                text=text.strip(),
                language=language,
                instruct=voice_description.strip(),
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
        meta = {"text": text.strip(), "language": language, "description": voice_description.strip()}
        return (
            (sr, _to_int16(wavs[0])),
            f"Voice Design erfolgreich generiert ({time.time() - start:.1f}s).",
            meta,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Voice Design] Fehler: {type(e).__name__}: {e}", flush=True)
        return None, f"Fehler: {type(e).__name__}: {e}", None


def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, progress=gr.Progress()):
    if not target_text or not target_text.strip():
        return None, "Fehler: Zieltext ist erforderlich."
    if not ref_audio:
        return None, "Fehler: Referenz-Audio ist erforderlich."
    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "Fehler: Referenztext ist erforderlich, wenn 'Nur x-Vector' nicht aktiviert ist."
    try:
        wav, ref_sr = _load_ref_audio(ref_audio)
        _log_request(
            "Voice Clone", sprache=language, referenz=f"{Path(ref_audio).name} ({len(wav) / ref_sr:.1f}s)",
            nur_xvector=use_xvector_only, text=target_text.strip()[:80],
        )
        with _lock:
            model = get_model("Base", progress)
            progress(None, desc="Generiere Audio ...")
            start = time.time()
            wavs, sr = model.generate_voice_clone(
                text=target_text.strip(),
                language=language,
                ref_audio=(wav, ref_sr),
                ref_text=ref_text.strip() if ref_text else None,
                x_vector_only_mode=use_xvector_only,
                max_new_tokens=2048,
            )
        return (sr, _to_int16(wavs[0])), f"Voice Clone erfolgreich generiert ({time.time() - start:.1f}s)."
    except Exception as e:  # noqa: BLE001
        print(f"[Voice Clone] Fehler: {type(e).__name__}: {e}", flush=True)
        return None, f"Fehler: {type(e).__name__}: {e}"


def generate_custom_voice(text, language, speaker, instruct, progress=gr.Progress()):
    if not text or not text.strip():
        return None, "Fehler: Text ist erforderlich."
    if not speaker:
        return None, "Fehler: Sprecher ist erforderlich."
    instruct = instruct.strip() if instruct else ""
    try:
        _log_request("TTS", sprache=language, sprecher=speaker, stilanweisung=instruct or None, text=text.strip()[:80])
        with _lock:
            model = get_model("CustomVoice", progress)
            progress(None, desc="Generiere Audio ...")
            start = time.time()
            wavs, sr = model.generate_custom_voice(
                text=text.strip(),
                language=language,
                speaker=speaker.lower().replace(" ", "_"),
                instruct=instruct or None,
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
        info = f"Stilanweisung: „{instruct}“" if instruct else "ohne Stilanweisung"
        return (sr, _to_int16(wavs[0])), f"Erfolgreich generiert ({time.time() - start:.1f}s, {info})."
    except Exception as e:  # noqa: BLE001
        print(f"[TTS] Fehler: {type(e).__name__}: {e}", flush=True)
        return None, f"Fehler: {type(e).__name__}: {e}"


KARTOFFEL_VARIANTS = {"Natural (19 Sprecher)": "KartoffelNatural", "Synthetic (4 Sprecher + Emotionen)": "KartoffelSynthetic"}


def _kartoffel_variant_changed(variant):
    synthetic = KARTOFFEL_VARIANTS[variant] == "KartoffelSynthetic"
    speakers = kartoffel_tts.SYNTHETIC_SPEAKERS if synthetic else kartoffel_tts.NATURAL_SPEAKERS
    return gr.update(choices=speakers, value=speakers[0]), gr.update(visible=synthetic)


def generate_kartoffel(text, variant, speaker, emotion, temperature, top_p, repetition_penalty,
                       progress=gr.Progress()):
    if not text or not text.strip():
        return None, "Fehler: Text ist erforderlich."
    model_type = KARTOFFEL_VARIANTS.get(variant)
    if model_type is None or not speaker:
        return None, "Fehler: Variante und Sprecher sind erforderlich."
    speakers = kartoffel_tts.SYNTHETIC_SPEAKERS if model_type == "KartoffelSynthetic" else kartoffel_tts.NATURAL_SPEAKERS
    if speaker not in speakers:
        return None, f"Fehler: Sprecher „{speaker}“ gibt es bei dieser Variante nicht. Verfügbar: {', '.join(speakers)}."
    emotion = emotion if model_type == "KartoffelSynthetic" and emotion and emotion != "Neutral" else None
    try:
        _log_request("Kartoffel", variante=MODEL_LABELS[model_type], sprecher=speaker, emotion=emotion,
                     text=text.strip()[:80])
        with _lock:
            model = get_model(model_type, progress)
            start = time.time()

            last_update = [0.0]

            def on_progress(chunk, chunks, tokens, max_tokens):
                now = time.time()
                if now - last_update[0] < 0.3:
                    return
                last_update[0] = now
                part = f"Abschnitt {chunk + 1}/{chunks} · " if chunks > 1 else ""
                progress((chunk + min(tokens / max_tokens, 0.99)) / chunks,
                         desc=f"Generiere Audio ... {part}~{tokens / 82:.0f}s Audio")

            wav, sr = model.generate(
                text.strip(), speaker, emotion,
                temperature=float(temperature), top_p=float(top_p),
                repetition_penalty=float(repetition_penalty), progress=on_progress,
            )
        elapsed = time.time() - start
        return (sr, _to_int16(wav)), (
            f"Erfolgreich generiert ({elapsed:.1f}s für {len(wav) / sr:.1f}s Audio, "
            f"{speaker}{f' – {emotion}' if emotion else ''})."
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Kartoffel] Fehler: {type(e).__name__}: {e}", flush=True)
        return None, f"Fehler: {type(e).__name__}: {e}"


_PARLER_QUALITY = "The recording is of very high quality, with the speaker's voice sounding clear and very close up."
PARLER_PRESETS = {
    "Nicole – ruhig und klar": f"Nicole's voice is calm and clear, with a moderate speed and pitch. {_PARLER_QUALITY}",
    "Michelle – lebhaft": f"Michelle speaks in an expressive and animated way, slightly fast, with a moderate pitch. {_PARLER_QUALITY}",
    "Christopher – sachlich": f"Christopher's voice is monotone yet slightly fast in delivery, with a low pitch. {_PARLER_QUALITY}",
    "Megan – sanft": f"Megan speaks softly and slowly, with a slightly high pitch. {_PARLER_QUALITY}",
    "Zufällige Frau – freundlich": "A female speaker delivers a slightly expressive and animated speech with a moderate "
                                   f"speed and pitch. {_PARLER_QUALITY}",
    "Zufälliger Mann – tief und langsam": f"A male speaker with a deep, low-pitched voice speaks slowly and calmly. {_PARLER_QUALITY}",
    "Zufälliger Mann – schnell, aufgeregt": "A male speaker speaks very fast with a high pitch and an excited, energetic "
                                            f"tone. {_PARLER_QUALITY}",
    "Hallige Aufnahme in großem Raum": "A female speaker speaks at a moderate speed in a very distant, echoing, "
                                       "reverberant room. Slightly noisy audio.",
}
PARLER_MAX_CHUNK_CHARS = 220


def generate_parler(text, description, seed, temperature, progress=gr.Progress()):
    import random

    if not text or not text.strip():
        return None, "Fehler: Text ist erforderlich.", None
    if not description or not description.strip():
        return None, "Fehler: Stimmbeschreibung ist erforderlich.", None
    try:
        seed = int(seed or 0)
    except (TypeError, ValueError):
        seed = 0
    if seed <= 0:
        seed = random.randint(1, 999_999)
    description = description.strip()
    try:
        _log_request("Parler", seed=seed, beschreibung=description, text=text.strip()[:80])
        with _lock:
            model = get_model("Parler", progress)
            chunks = kartoffel_tts.split_text(text.strip(), PARLER_MAX_CHUNK_CHARS)
            start = time.time()
            parts = []
            for i, chunk in enumerate(chunks):
                part = f" (Abschnitt {i + 1}/{len(chunks)})" if len(chunks) > 1 else ""
                progress(i / len(chunks), desc=f"Generiere Audio{part} ...")
                parts.append(model.generate(chunk, description, seed, float(temperature)))
                if i < len(chunks) - 1:
                    parts.append(np.zeros(int(model.sampling_rate * 0.25), dtype=np.float32))
            wav = np.concatenate(parts)
            sr = model.sampling_rate
        meta = {"text": text.strip(), "language": "German", "description": f"Parler-TTS: {description} (Seed {seed})"}
        return (sr, _to_int16(wav)), (
            f"Erfolgreich generiert ({time.time() - start:.1f}s für {len(wav) / sr:.1f}s Audio, Seed {seed}). "
            "Gleicher Seed + Text + Beschreibung ergibt exakt dasselbe Ergebnis."
        ), meta
    except Exception as e:  # noqa: BLE001
        print(f"[Parler] Fehler: {type(e).__name__}: {e}", flush=True)
        return None, f"Fehler: {type(e).__name__}: {e}", None


def _parse_seed(seed) -> int:
    import random

    try:
        seed = int(seed or 0)
    except (TypeError, ValueError):
        seed = 0
    return seed if seed > 0 else random.randint(1, 999_999)


def generate_audio8(text, ref_audio, ref_text, temperature, top_p, top_k, seed, progress=gr.Progress()):
    if not text or not text.strip():
        return None, "Fehler: Text ist erforderlich.", None
    if ref_audio and (not ref_text or not ref_text.strip()):
        return None, "Fehler: Zum Referenz-Audio wird der Referenztext (exaktes Transkript) benötigt.", None
    seed = _parse_seed(seed)
    try:
        ref = None
        if ref_audio:
            wav, ref_sr = _load_ref_audio(ref_audio)
            if len(wav) / ref_sr > 30:
                return None, "Fehler: Referenz-Audio bitte auf höchstens 30 Sekunden kürzen.", None
            ref = (wav, ref_sr)
        _log_request("Audio8", seed=seed, referenz=Path(ref_audio).name if ref_audio else None,
                     text=text.strip()[:80])
        with _lock:
            model = get_model("Audio8", progress)
            start = time.time()
            last_update = [0.0]

            def on_progress(chunk, chunks, tokens, max_tokens):
                now = time.time()
                if now - last_update[0] < 0.3:
                    return
                last_update[0] = now
                part = f"Abschnitt {chunk + 1}/{chunks} · " if chunks > 1 else ""
                progress((chunk + min(tokens / max_tokens, 0.99)) / chunks,
                         desc=f"Generiere Audio ... {part}~{tokens / audio8_tts.FRAMES_PER_SECOND:.0f}s Audio")

            wav, sr = model.generate(
                text.strip(), ref, ref_text, temperature=float(temperature), top_p=float(top_p),
                top_k=int(top_k), seed=seed, progress=on_progress,
            )
        voice = "geklonte Stimme" if ref else "zufällige Stimme"
        meta = {"text": text.strip(), "language": "German",
                "description": f"Audio8-TTS: {voice} (Seed {seed})"}
        return (sr, _to_int16(wav)), (
            f"Erfolgreich generiert ({time.time() - start:.1f}s für {len(wav) / sr:.1f}s Audio, {voice}, "
            f"Seed {seed})."
        ), meta
    except Exception as e:  # noqa: BLE001
        print(f"[Audio8] Fehler: {type(e).__name__}: {e}", flush=True)
        return None, f"Fehler: {type(e).__name__}: {e}", None


CHATTERBOX_MAX_CHUNK_CHARS = 250


def generate_chatterbox(text, ref_audio, exaggeration, cfg_weight, temperature, seed, progress=gr.Progress()):
    if not text or not text.strip():
        return None, "Fehler: Text ist erforderlich."
    seed = _parse_seed(seed)
    try:
        if ref_audio:
            _load_ref_audio(ref_audio)  # Format und Länge prüfen
        _log_request("Chatterbox", seed=seed, referenz=Path(ref_audio).name if ref_audio else None,
                     exaggeration=exaggeration, cfg=cfg_weight, text=text.strip()[:80])
        with _lock:
            model = get_model("Chatterbox", progress)
            chunks = kartoffel_tts.split_text(text.strip(), CHATTERBOX_MAX_CHUNK_CHARS)
            start = time.time()
            parts = []
            for i, chunk in enumerate(chunks):
                part = f" (Abschnitt {i + 1}/{len(chunks)})" if len(chunks) > 1 else ""
                progress(i / len(chunks), desc=f"Generiere Audio{part} ...")
                parts.append(model.generate(chunk, ref_audio or None, exaggeration, cfg_weight, temperature, seed + i))
                if i < len(chunks) - 1:
                    parts.append(np.zeros(int(model.sampling_rate * 0.2), dtype=np.float32))
            wav = np.concatenate(parts)
            sr = model.sampling_rate
        voice = "geklonte Stimme" if ref_audio else "Standardstimme"
        return (sr, _to_int16(wav)), (
            f"Erfolgreich generiert ({time.time() - start:.1f}s für {len(wav) / sr:.1f}s Audio, {voice}, Seed {seed})."
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Chatterbox] Fehler: {type(e).__name__}: {e}", flush=True)
        return None, f"Fehler: {type(e).__name__}: {e}"


def _transcribe(audio_path, choice, language, progress) -> tuple[str, str]:
    """Gibt (Text, Status) zurück. Wirft ValueError bei ungültiger Eingabe."""
    if not audio_path:
        raise ValueError("Bitte zuerst ein Audio hochladen oder aufnehmen.")
    model_type = ASR_CHOICES.get(choice, ASR_CHOICES[ASR_DEFAULT_CHOICE])
    lang = ASR_LANGUAGES.get(language, ASR_LANGUAGES[ASR_DEFAULT_LANGUAGE])
    wav, sr = _load_ref_audio(audio_path)
    duration = len(wav) / sr
    _log_request("ASR", modell=MODEL_LABELS[model_type], sprache=lang or "auto",
                 audio=f"{Path(audio_path).name} ({duration:.1f}s)")
    with _lock:
        model = get_model(model_type, progress)
        start = time.time()

        def on_progress(i, n):
            part = f" (Abschnitt {i + 1}/{n})" if n > 1 else ""
            progress(i / n, desc=f"Erkenne Sprache{part} ...")

        text = model.transcribe(wav, sr, progress=on_progress, language=lang)
        detected = getattr(model, "last_language", None)
    detected = f", erkannte Sprache: {detected}" if detected else ""
    status = (f"Erkannt mit {MODEL_LABELS[model_type]} ({time.time() - start:.1f}s für {duration:.1f}s Audio"
              f"{detected}). Bitte prüfen und bei Bedarf korrigieren.")
    if not text:
        status = "Keine Sprache erkannt."
    return text, status


def transcribe_audio(audio_path, choice, language, progress=gr.Progress()):
    try:
        return _transcribe(audio_path, choice, language, progress)
    except Exception as e:  # noqa: BLE001
        print(f"[ASR] Fehler: {type(e).__name__}: {e}", flush=True)
        return "", f"Fehler: {type(e).__name__}: {e}"


def transcribe_reference(audio_path, choice, language, current_text, progress=gr.Progress()):
    """Füllt ein Referenztext-Feld; bei Fehlern bleibt der bisherige Text erhalten."""
    try:
        text, status = _transcribe(audio_path, choice, language, progress)
        return (text or current_text), status
    except Exception as e:  # noqa: BLE001
        print(f"[ASR] Fehler: {type(e).__name__}: {e}", flush=True)
        return current_text, f"Transkript-Fehler: {type(e).__name__}: {e}"


def _model_panel_updates():
    return [model_table_html(), status_note_markdown()]


def _make_toggle_handler(model_type: str):
    label = MODEL_LABELS[model_type]

    def handler(progress=gr.Progress()):
        try:
            if model_type in _loaded:
                unload_model(model_type)
                msg = f"{label} entladen."
            else:
                evicted_before = set(_loaded)
                get_model(model_type, progress)
                evicted = evicted_before - set(_loaded)
                msg = f"{label} ist geladen."
                if evicted:
                    msg += " Entladen (Speicherlimit): " + ", ".join(MODEL_LABELS[m] for m in evicted)
        except Exception as e:  # noqa: BLE001
            msg = f"Fehler: {type(e).__name__}: {e}"
        return [msg, *_model_panel_updates()]

    return handler


def _unload_all_handler():
    done = unload_all()
    msg = ("Entladen: " + ", ".join(MODEL_LABELS[m] for m in done)) if done else "Es war kein Modell geladen."
    return [msg, *_model_panel_updates()]


# ---------------------------------------------------------------- Installation (Download & Install)
_installing: dict = {}  # Modell-Schlüssel -> {"progress": 0..1, "text": str} (wartend oder laufend)
_install_errors: dict = {}  # Modell-Schlüssel -> Fehlermeldung der letzten Installation
_install_queue: list = []
_install_cv = threading.Condition()
_install_thread: threading.Thread | None = None
GATED_HINT = ("Dieses Modell ist zugangsbeschränkt: auf Hugging Face einloggen, auf der Modellseite die "
              "Bedingungen akzeptieren und unter „Modelle verwalten“ einen Hugging-Face-Token eintragen.")


def request_install(key: str) -> str:
    """Stellt ein Modell in die Installations-Warteschlange (Downloads laufen nacheinander im Hintergrund)."""
    global _install_thread
    label = MODEL_LABELS[key]
    with _install_cv:
        if key in _installing:
            return f"{label} wird bereits installiert."
        if key in _uninstalling:
            return f"{label} wird gerade deinstalliert – bitte warten."
        if is_installed(key):
            return f"{label} ist bereits installiert."
        _install_errors.pop(key, None)
        _installing[key] = {"progress": 0.0, "text": "wartet ..."}
        _install_queue.append(key)
        _install_cv.notify()
        if _install_thread is None or not _install_thread.is_alive():
            _install_thread = threading.Thread(target=_install_loop, daemon=True, name="model-install")
            _install_thread.start()
    print(f"[Installation] {label} in die Warteschlange gestellt.", flush=True)
    return f"{label}: Installation gestartet – Fortschritt in der Modell-Übersicht."


def _install_loop() -> None:
    while True:
        with _install_cv:
            if not _install_queue:
                return
            key = _install_queue[0]
        try:
            _install_model(key)
            print(f"[Installation] {MODEL_LABELS[key]} ist installiert.", flush=True)
        except Exception as e:  # noqa: BLE001
            msg = str(e) or type(e).__name__
            if "GatedRepo" in type(e).__name__ or "401" in msg or "403" in msg:
                msg = GATED_HINT
            _install_errors[key] = msg
            print(f"[Installation] Fehler bei {MODEL_LABELS[key]}: {type(e).__name__}: {e}", flush=True)
        finally:
            for k in MODEL_LABELS:
                MODEL_DISK_GB[k] = _dir_size_gb(MODEL_PATHS[k]) if is_installed(k) else 0.0
            with _install_cv:
                _install_queue.pop(0)
                _installing.pop(key, None)


def _install_model(key: str) -> None:
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    label = MODEL_LABELS[key]
    parts = missing_parts(key)
    weights = [SIDE_ENV_DOWNLOAD_GB if k == "env" else max(i.approx_gb, 0.01) if k == "repo" else 0.03
               for k, i in parts]
    total = sum(weights) or 1.0
    done = 0.0
    _check_disk_space(parts)
    started = time.time()
    print(f"[Installation] {label}: {len(parts)} Teil(e), ca. {_fmt_size(download_size_gb(key))} ...", flush=True)
    for (kind, item), weight in zip(parts, weights):
        def report(frac, text, detail="", busy=False, base=done, weight=weight):
            _installing[key] = {"progress": (base + weight * min(max(frac, 0.0), 1.0)) / total, "text": text,
                                "detail": detail, "busy": busy, "started": started}

        if kind == "env":
            _install_env(item, report)
        elif kind == "repo":
            _install_repo(item, report)
        else:
            _install_deepfilter(item, report)
        done += weight


def _install_repo(repo, report) -> None:
    name = repo.repo_id.split("/")[1]
    report(0.0, f"{name}: prüfe Dateien ...")
    repo.local_dir.mkdir(parents=True, exist_ok=True)
    marker = repo.local_dir / INSTALLING_MARKER
    marker.write_text("download in progress\n", encoding="utf-8")
    info = model_updates.check(repo)

    def on_progress(loaded, total_bytes, filename):
        report(loaded / total_bytes, f"{name}: {loaded / total_bytes * 100:.0f} % "
                                     f"({model_updates.format_bytes(loaded)} / {model_updates.format_bytes(total_bytes)})",
               detail=str(filename or ""))

    print(f"[Installation] Lade {repo.repo_id} ({model_updates.format_bytes(info.total_bytes)}) ...", flush=True)
    model_updates.download(info, on_progress)
    marker.unlink(missing_ok=True)


def _install_env(env: str, report) -> None:
    """Richtet eine Zusatz-Umgebung über setup_env.py ein (gleiche Logik wie der Installer)."""
    import subprocess

    log_file = BASE_DIR / f"install_{env}.log"
    cmd = [sys.executable, str(BASE_DIR / "setup_env.py"), "--env", env,
           "--backend", app_config.torch_backend(), "--non-interactive"]
    print(f"[Installation] Richte Python-Umgebung '{env}' ein (Protokoll: {log_file.name}) ...", flush=True)
    report(0.0, f"Python-Umgebung '{env}' wird vorbereitet ...")
    env_vars = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    step = [0, 4, "Start"]
    detail = ""
    error = None
    with log_file.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", env=env_vars, cwd=str(BASE_DIR),
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for line in proc.stdout:
            log.write(line)
            log.flush()
            line = line.strip()
            if line.startswith("##STEP"):
                nums, _, text = line[7:].partition(" ")
                i, _, n = nums.partition("/")
                step = [int(i), int(n), text]
                detail = ""
                print(f"[Installation] {env}: Schritt {i}/{n} – {text}", flush=True)
            elif line.startswith("##ERROR"):
                error = line[8:]
            elif line and not line.startswith("##"):
                detail = line[:200]  # letzte Ausgabe von uv/pip, z. B. "Downloading torch (2.3GiB)"
            frac = (step[0] - 1 + 0.5) / step[1] if step[0] else 0.0
            report(frac, f"Python-Umgebung '{env}': Schritt {step[0]}/{step[1]} – {step[2]}", detail=detail,
                   busy=True)
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Einrichtung der Umgebung '{env}' fehlgeschlagen: {error or 'siehe ' + log_file.name}")


def _install_deepfilter(target: Path, report) -> None:
    import urllib.request

    url, _ = audio_cleanup.deepfilter_asset()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"[Installation] Lade {url} ...", flush=True)
    for attempt in range(1, 7):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as f:
                size = int(resp.headers.get("Content-Length") or 0)
                loaded = 0
                while chunk := resp.read(256 * 1024):
                    f.write(chunk)
                    loaded += len(chunk)
                    if size:
                        report(loaded / size, f"deep-filter: {loaded / size * 100:.0f} %")
            break
        except OSError:
            if attempt == 6:
                raise
            time.sleep(3 * attempt)
    tmp.replace(target)
    if os.name != "nt":
        target.chmod(0o755)


def _make_install_handler(key: str):
    def handler():
        return [request_install(key), *_model_panel_updates()]

    return handler


# ---------------------------------------------------------------- Deinstallation
_uninstalling: dict = {}  # Modell-Schlüssel -> Startzeit


def _inside(path: Path, root: Path) -> bool:
    try:
        path, root = path.resolve(), root.resolve()
    except OSError:
        return False
    return path != root and path.is_relative_to(root)


def _uninstall_targets(key: str) -> list:
    """Ordner, die beim Deinstallieren gelöscht werden. Teile, die ein anderes installiertes Modell mitbenutzt
    (z. B. SNAC bei beiden Kartoffel-Modellen), bleiben erhalten."""
    def used_elsewhere(users) -> bool:
        return any(k != key and (k in _installing or is_installed(k)) for k in users)

    targets = [r.local_dir for r in MODEL_REPOS[key] if r.local_dir.exists() and not used_elsewhere(r.used_by)]
    env = MODEL_ENVS.get(key)
    if env and not used_elsewhere([k for k, e in MODEL_ENVS.items() if e == env]):
        targets.append(BASE_DIR / app_config.SIDE_ENVS[env])
    if key == "DeepFilterNet":
        targets.append(DEEPFILTER_EXE.parent)
    # Nur Modellordner und eigene Zusatz-Umgebungen (.venv-*) – niemals etwas anderes.
    return [t for t in dict.fromkeys(targets) if t.exists() and (
        _inside(t, MODELS_DIR) or (t.parent.resolve() == BASE_DIR.resolve() and t.name.startswith(".venv-")))]


def _disk_usage(path: Path) -> tuple:
    """(wirklich belegte Bytes, im Explorer angezeigte Bytes). Per Hardlink geteilte Dateien belegen keinen
    eigenen Platz und werden beim Löschen auch nicht frei."""
    real = shown = 0
    for f in path.rglob("*"):
        try:
            st = f.lstat()
        except OSError:
            continue
        if f.is_file():
            shown += st.st_size
            real += st.st_size if st.st_nlink <= 1 else 0
    return real, shown


def _rmtree(path: Path) -> None:
    import shutil
    import stat

    def force(func, p, exc):
        if isinstance(exc, FileNotFoundError):
            return
        os.chmod(p, stat.S_IWRITE)  # schreibgeschützte Dateien (Windows)
        func(p)

    for attempt in range(6):
        try:
            shutil.rmtree(path, onexc=force)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == 5:
                raise
            time.sleep(1 + attempt)  # Hilfsprozesse geben Dateien unter Windows erst kurz nach dem Beenden frei


def uninstall_model(key: str) -> str:
    """Löscht Modelldateien (und ggf. die eigene Python-Umgebung) eines Modells."""
    label = MODEL_LABELS[key]
    with _install_cv:
        if key in _installing:
            return f"{label} wird gerade installiert – bitte warten oder danach deinstallieren."
        if key in _updates:
            return f"Für {label} läuft gerade ein Update – bitte danach deinstallieren."
        if key in _uninstalling:
            return f"{label} wird bereits deinstalliert."
        targets = _uninstall_targets(key)
        if not targets:
            return f"{label} ist nicht installiert."
        _uninstalling[key] = time.time()
    print(f"[Deinstallation] {label}: lösche {', '.join(str(t) for t in targets)} ...", flush=True)
    try:
        unload_model(key)  # wartet ggf. auf eine laufende Generierung
        if key == "Clear":
            _clear_models.clear()
            gc.collect()
        real = shown = 0
        for target in targets:
            r, s = _disk_usage(target)
            real, shown = real + r, shown + s
        for target in targets:  # bricht das Löschen ab, gilt das Modell trotzdem als nicht installiert
            with contextlib.suppress(OSError):
                (target / INSTALLING_MARKER).write_text("uninstall in progress\n", encoding="utf-8")
        errors = []
        for target in targets:
            try:
                _rmtree(target)
            except OSError as e:
                errors.append(f"{target.name}: {e}")
            parent = target.parent
            while _inside(parent, MODELS_DIR):  # leere Überordner (z. B. clearvoice/checkpoints) entfernen
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        if MODEL_ENVS.get(key) and not _installing:
            with contextlib.suppress(Exception):
                import setup_env

                real += setup_env.prune_cache()  # nur noch von dieser Umgebung benutzte Pakete im uv-Cache
        if errors:
            print(f"[Deinstallation] {label}: nicht alles gelöscht: {'; '.join(errors)}", flush=True)
            return (f"{label}: nicht alle Dateien konnten gelöscht werden ({'; '.join(errors)}). Bitte die App neu "
                    "starten und erneut deinstallieren.")
        msg = f"{label} deinstalliert – {_fmt_size(real / 1024**3)} freigegeben."
        if shown > real + 0.1 * 1024**3:
            msg += (f" (Der Explorer zeigte {_fmt_size(shown / 1024**3)} an; der Rest sind Dateien, die mit anderen "
                    "Umgebungen geteilt werden.)")
        print(f"[Deinstallation] {msg}", flush=True)
        return msg
    finally:
        _install_errors.pop(key, None)
        for k in MODEL_LABELS:
            MODEL_DISK_GB[k] = _dir_size_gb(MODEL_PATHS[k]) if is_installed(k) else 0.0
        _uninstalling.pop(key, None)


def _make_uninstall_handler(key: str):
    def handler():
        try:
            msg = uninstall_model(key)
        except Exception as e:  # noqa: BLE001
            msg = f"Fehler beim Deinstallieren: {type(e).__name__}: {e}"
        return [msg, *_model_panel_updates()]

    return handler


def hf_token_status() -> str:
    from huggingface_hub import get_token

    return ("✅ Ein Hugging-Face-Token ist gespeichert." if get_token()
            else "Kein Hugging-Face-Token gespeichert (nur für zugangsbeschränkte Modelle wie Kartoffel nötig).")


def save_hf_token(token: str):
    from huggingface_hub import login

    token = (token or "").strip()
    if not token:
        return gr.update(), "Bitte einen Token eingeben (https://huggingface.co/settings/tokens, Typ „Read“)."
    try:
        login(token=token, add_to_git_credential=False)
        return "", "✅ Token gespeichert."
    except Exception as e:  # noqa: BLE001
        return gr.update(), f"Fehler: Token ungültig ({type(e).__name__})."


# ---------------------------------------------------------------- Audio-Cleanup
_clear_models: dict = {}
CLEARVOICE_MODEL_NAMES = {label: name for label, (name, _) in CLEARVOICE_MODELS.items()}


def _cleanup_audio_path(path):
    if not path:
        raise ValueError("Bitte zuerst ein Audio hochladen oder aufnehmen.")
    ext = Path(path).suffix.lower()
    if ext and ext not in SUPPORTED_AUDIO_EXTS:
        raise ValueError(f"Format '{ext}' wird nicht unterstützt. Erlaubt: WAV, MP3, FLAC, OGG/Opus, AIFF, CAF.")
    return path


def _require_installed(key: str) -> None:
    if not is_installed(key):
        raise FileNotFoundError(f"{MODEL_LABELS[key]} ist nicht installiert – bitte über „⬇ Installieren“ "
                                "herunterladen.")


def clean_clear(audio_path, variant, strength, normalize, progress=gr.Progress()):
    try:
        _cleanup_audio_path(audio_path)
        _require_installed("Clear")
        file = audio_cleanup.CLEAR_VARIANTS[CLEAR_VARIANT_CHOICES[variant]]
        _log_request("Clear", variante=variant, staerke=strength, datei=Path(audio_path).name)
        progress(0.0, desc="Lese Audio ...")
        wav = audio_cleanup.load_audio_48k(audio_path)
        start = time.time()
        model = _clear_models.get(file)
        if model is None:
            model = _clear_models[file] = audio_cleanup.ClearDenoiser(MODEL_PATHS["Clear"] / file)
        atten = 100.0 if strength >= 100 else 6 + strength / 100 * 40
        out = model.enhance(wav, atten_lim_db=atten,
                            progress=lambda i, n: progress(i / n, desc="Entferne Störgeräusche ..."))
        out = audio_cleanup.finalize(out, normalize)
        return (audio_cleanup.SR, _to_int16(out)), (
            f"Fertig ({time.time() - start:.1f}s für {len(wav) / audio_cleanup.SR:.1f}s Audio, 48 kHz).")
    except Exception as e:  # noqa: BLE001
        print(f"[Clear] Fehler: {type(e).__name__}: {e}", flush=True)
        return None, f"Fehler: {type(e).__name__}: {e}"


def clean_deepfilter(audio_path, atten_db, post_filter, normalize, progress=gr.Progress()):
    try:
        _cleanup_audio_path(audio_path)
        _require_installed("DeepFilterNet")
        _log_request("DeepFilterNet", absenkung=atten_db, nachfilter=post_filter, datei=Path(audio_path).name)
        progress(0.0, desc="Lese Audio ...")
        wav = audio_cleanup.load_audio_48k(audio_path)
        progress(None, desc="Entferne Störgeräusche (DeepFilterNet) ...")
        start = time.time()
        out = audio_cleanup.run_deepfilter(DEEPFILTER_EXE, wav, atten_lim_db=100.0 if atten_db >= 100 else atten_db,
                                           post_filter=post_filter)
        out = audio_cleanup.finalize(out, normalize)
        return (audio_cleanup.SR, _to_int16(out)), (
            f"Fertig ({time.time() - start:.1f}s für {len(wav) / audio_cleanup.SR:.1f}s Audio, 48 kHz).")
    except Exception as e:  # noqa: BLE001
        print(f"[DeepFilterNet] Fehler: {type(e).__name__}: {e}", flush=True)
        return None, f"Fehler: {type(e).__name__}: {e}"


def clean_clearvoice(audio_path, model_label, normalize, progress=gr.Progress()):
    import soundfile as sf
    import tempfile

    try:
        _cleanup_audio_path(audio_path)
        name = CLEARVOICE_MODEL_NAMES[model_label]
        _log_request("ClearVoice", modell=name, datei=Path(audio_path).name)
        progress(0.0, desc="Lese Audio ...")
        wav, sr = librosa.load(audio_path, sr=None, mono=True)
        with _lock:
            client = get_model("ClearVoice", progress)
            progress(None, desc=f"Verarbeite mit {name} (beim ersten Mal wird das Netz geladen) ...")
            start = time.time()
            fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="clearvoice_in_")
            os.close(fd)
            try:
                sf.write(tmp, wav, sr)
                out, out_sr = client.process(name, tmp)
            finally:
                Path(tmp).unlink(missing_ok=True)
            _model_sizes_gb["ClearVoice"] = client.vram_gb
            _remember_vram("ClearVoice", client.vram_gb)
        out = np.asarray(out, dtype=np.float32)
        tracks = [out] if out.ndim == 1 else list(out)
        tracks = [(out_sr, _to_int16(audio_cleanup.finalize(t, normalize))) for t in tracks]
        second = tracks[1] if len(tracks) > 1 else None
        msg = (f"Fertig ({time.time() - start:.1f}s für {len(wav) / sr:.1f}s Audio, {out_sr // 1000} kHz"
               + (", 2 Sprecher getrennt" if second else "") + ").")
        return tracks[0], gr.update(value=second, visible=second is not None), msg
    except Exception as e:  # noqa: BLE001
        print(f"[ClearVoice] Fehler: {type(e).__name__}: {e}", flush=True)
        return None, gr.update(value=None, visible=False), f"Fehler: {type(e).__name__}: {e}"


GPU_EVENT = {"concurrency_id": "gpu", "concurrency_limit": 1}

VOICES_DIR = BASE_DIR / "voices"
VOICE_AUDIO_NAME = "referenz.wav"
VOICE_META_NAME = "stimme.json"


def _safe_voice_name(name: str) -> str:
    cleaned = "".join(c for c in (name or "").strip() if c.isalnum() or c in " -_äöüÄÖÜß").strip()
    return cleaned[:60]


def list_voices() -> list:
    if not VOICES_DIR.exists():
        return []
    return sorted(
        (d.name for d in VOICES_DIR.iterdir() if (d / VOICE_AUDIO_NAME).exists() and (d / VOICE_META_NAME).exists()),
        key=str.lower,
    )


def _voice_dropdown_update(selected=None):
    voices = list_voices()
    return gr.update(choices=voices, value=selected if selected in voices else None)


def _save_voice(name: str, sr: int, wav, ref_text: str, language: str, description: str, source: str):
    import soundfile as sf

    safe = _safe_voice_name(name)
    if not safe:
        raise ValueError("Bitte einen Namen für die Stimme eingeben.")
    folder = VOICES_DIR / safe
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(str(folder / VOICE_AUDIO_NAME), np.asarray(wav), int(sr))
    meta = {
        "name": safe,
        "ref_text": ref_text,
        "language": language,
        "description": description,
        "source": source,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (folder / VOICE_META_NAME).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Stimmen] '{safe}' gespeichert ({folder}).", flush=True)
    return safe


def save_design_voice(name, audio, last_meta, source="Voice Design"):
    if audio is None or not last_meta:
        return "Fehler: Zuerst eine Stimme generieren.", _voice_dropdown_update()
    try:
        sr, wav = audio
        safe = _save_voice(
            name, sr, wav, last_meta["text"], last_meta["language"], last_meta["description"], source,
        )
        return f"Stimme „{safe}“ gespeichert – nutzbar in den Klon-Tabs (Voice Clone, Audio8, Chatterbox).", \
            _voice_dropdown_update(safe)
    except Exception as e:  # noqa: BLE001
        return f"Fehler: {e}", _voice_dropdown_update()


def save_parler_voice(name, audio, last_meta):
    return save_design_voice(name, audio, last_meta, source="Parler-TTS")


def save_audio8_voice(name, audio, last_meta):
    return save_design_voice(name, audio, last_meta, source="Audio8-TTS")


def save_clone_reference(name, ref_audio, ref_text, language):
    if not ref_audio:
        return "Fehler: Kein Referenz-Audio vorhanden.", _voice_dropdown_update()
    if not ref_text or not ref_text.strip():
        return "Fehler: Für eine gespeicherte Stimme wird der Referenztext benötigt.", _voice_dropdown_update()
    try:
        wav, sr = _load_ref_audio(ref_audio)
        safe = _save_voice(name, sr, wav, ref_text.strip(), language, "", "Voice Clone (Referenz)")
        return f"Stimme „{safe}“ gespeichert.", _voice_dropdown_update(safe)
    except Exception as e:  # noqa: BLE001
        return f"Fehler: {e}", _voice_dropdown_update()


def load_voice(name):
    if not name:
        return gr.update(), gr.update(), gr.update(), gr.update(), ""
    folder = VOICES_DIR / name
    try:
        meta = json.loads((folder / VOICE_META_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return gr.update(), gr.update(), gr.update(), gr.update(), f"Fehler beim Laden: {e}"
    info = f"Stimme „{name}“ geladen ({meta.get('source', '?')}, {meta.get('created', '')})."
    if meta.get("description"):
        info += f" · Beschreibung: {meta['description']}"
    language = meta.get("language") if meta.get("language") in LANGUAGES else gr.update()
    return str(folder / VOICE_AUDIO_NAME), meta.get("ref_text", ""), False, language, info


def _voice_choices_update():
    """Aktualisiert nur die Auswahlliste, die aktuelle Auswahl bleibt erhalten."""
    return gr.update(choices=list_voices())


def load_voice_reference(name):
    """Für Audio8/Chatterbox: (Audio-Pfad, Referenztext, Statusmeldung)."""
    if not name:
        return gr.update(), gr.update(), ""
    folder = VOICES_DIR / name
    try:
        meta = json.loads((folder / VOICE_META_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return gr.update(), gr.update(), f"Fehler beim Laden: {e}"
    info = f"Stimme „{name}“ geladen ({meta.get('source', '?')})."
    return str(folder / VOICE_AUDIO_NAME), meta.get("ref_text", ""), info


def delete_voice(name):
    import shutil

    if not name:
        return "Keine Stimme ausgewählt.", _voice_dropdown_update()
    folder = VOICES_DIR / name
    if folder.parent == VOICES_DIR and folder.exists():
        shutil.rmtree(folder)
        print(f"[Stimmen] '{name}' gelöscht.", flush=True)
        return f"Stimme „{name}“ gelöscht.", _voice_dropdown_update()
    return "Stimme nicht gefunden.", _voice_dropdown_update()


def _hf_button(model_type: str, text: str | None = None):
    return gr.Button(
        text or "🤗 Modell auf Hugging Face", link=MODEL_URLS[model_type], link_target="_blank",
        size="sm", variant="secondary", scale=0, min_width=230, elem_classes="hf-link",
    )


def _tab_header(title: str, hint: str, *model_types, texts=()):
    """Überschrift + Kurzbeschreibung links, Hugging-Face-Link(s) rechts. Gibt die Link-Buttons zurück."""
    with gr.Row(elem_classes="tab-header"):
        with gr.Column(elem_classes="tab-header-text"):
            gr.Markdown(f"### {title}")
            gr.Markdown(hint, elem_classes="hint")
        buttons = [_hf_button(key, texts[i] if i < len(texts) else None) for i, key in enumerate(model_types)]
    return buttons


def _section(title: str):
    gr.Markdown(f"#### {title}", elem_classes="section-title")


# Overlays über Modell-Bereichen, deren Modell noch fehlt: {"col", "md", "btn", "key", "source"}
_OVERLAYS: list = []


def _overlay_key(entry, value) -> str:
    key = entry["key"]
    return key if isinstance(key, str) else key(value)


@contextlib.contextmanager
def _model_area(key, source=None):
    """Bereich mit Installations-Overlay. key: Modell-Schlüssel oder Funktion(Wert von source) -> Schlüssel."""
    with gr.Column(elem_classes="model-area"):
        with gr.Column(visible=False, elem_classes="install-overlay") as col:
            with gr.Column(elem_classes="install-card"):
                md = gr.Markdown()
                btn = gr.Button("⬇ Modell herunterladen & installieren", variant="primary",
                                elem_classes="install-btn")
        _OVERLAYS.append({"col": col, "md": md, "btn": btn, "key": key, "source": source})
        yield


def _overlay_state(key: str) -> list:
    label = MODEL_LABELS[key]
    if key in _installing:
        state = _installing[key]
        detail = f"\n\n`{state['detail']}`" if state.get("detail") else ""
        return [gr.update(visible=True),
                f"### ⏳ {label} wird installiert\n**{state.get('progress', 0) * 100:.0f} %** – "
                f"{state.get('text', '')}{detail}\n\nDu kannst währenddessen andere Tabs nutzen "
                "(Fortschritt auch oben unter „GPU & Hintergrund-Aufgaben“).",
                gr.update(value="⏳ Installation läuft ...", interactive=False)]
    if key in _install_errors:
        return [gr.update(visible=True),
                f"### ❌ Installation von {label} fehlgeschlagen\n{_install_errors[key]}",
                gr.update(value="⬇ Erneut versuchen", interactive=True)]
    if is_installed(key):
        return [gr.update(visible=False), gr.skip(), gr.skip()]
    env = side_env_note(key)
    env = f"\n\n⚠️ {env}" if env else ""
    return [gr.update(visible=True),
            f"### ⬇️ {label} ist noch nicht installiert\nDownload ca. **{_fmt_size(download_size_gb(key))}**, "
            f"gespeichert in `{MODELS_DIR}`.{env}",
            gr.update(value="⬇ Modell herunterladen & installieren", interactive=True)]


def _hint(text: str):
    return gr.Markdown(text, elem_classes="hint")


def _gen_button(text: str = "🔊 Generieren"):
    return gr.Button(text, variant="primary", elem_classes="gen-btn")


def _status_box():
    return gr.Textbox(
        show_label=False, container=False, lines=1, max_lines=1, interactive=False,
        placeholder="Status", elem_classes="status-line",
    )


def _output_audio(label: str = "Generiertes Audio"):
    return gr.Audio(label=label, type="numpy")


def _input_audio(label: str = "Audio"):
    return gr.Audio(label=label, type="filepath", sources=["upload", "microphone"])


CLEAR_VARIANT_CHOICES = {"Studio – maximal sauber": "studio", "Natural – natürlicher Klang": "natural"}
CLEANUP_FORMATS_HINT = FORMATS_HINT.split(" Empfehlung:")[0] + " Ausgabe als 48-kHz-Audio (Mono)."


def _clearvoice_choice_changed(label):
    name, desc = CLEARVOICE_MODELS[label]
    separation = name.endswith("_SS_16K")
    return desc, gr.update(label="Sprecher 1" if separation else "Bereinigtes Audio")


def _text_input(value: str = "", placeholder: str = "Text eingeben, der gesprochen werden soll ..."):
    return gr.Textbox(label="Text zum Sprechen", lines=4, placeholder=placeholder, value=value)


def _temperature(minimum: float, maximum: float, value: float):
    return gr.Slider(minimum, maximum, value=value, step=0.05, label="Temperatur",
                     info="Höher = lebendiger, aber weniger stabil")


def _seed(info: str | None = None):
    return gr.Number(label="Seed (0 = zufällig)", value=0, precision=0, minimum=0,
                     info=info or "Gleicher Seed + gleiche Eingaben = identisches Ergebnis")


TOP_P_INFO = ("Pro Schritt nur aus den wahrscheinlichsten Möglichkeiten wählen, die zusammen diesen Anteil "
              "ausmachen. Niedriger = gleichmäßiger, höher = variabler")


def _save_voice_box(hint: str):
    """Zum Speichern einer Stimme: Name + Button. Gibt (Name, Button) zurück."""
    with gr.Accordion("💾 Stimme speichern", open=True):
        _hint(hint)
        with gr.Row(elem_classes="voice-row"):
            name = gr.Textbox(show_label=False, container=False, placeholder="Name der Stimme", scale=3)
            button = gr.Button("💾 Speichern", scale=0, min_width=170, elem_classes="voice-action")
    return name, button


def _saved_voice_controls(info: str = "Füllt Referenz-Audio und Referenztext automatisch aus"):
    """Auswahl gespeicherter Stimmen + gleich breite Buttons Aktualisieren/Löschen."""
    select = gr.Dropdown(label="Gespeicherte Stimme", info=info, choices=list_voices(), value=None,
                         allow_custom_value=True)
    with gr.Row(elem_classes="btn-pair"):
        refresh = gr.Button("🔄 Aktualisieren", size="sm", min_width=0)
        delete = gr.Button("🗑 Löschen", size="sm", variant="stop", min_width=0)
    return select, refresh, delete


def _asr_language_visible(choice):
    return gr.update(visible=ASR_CHOICES.get(choice) not in AUDIO8_ASR_MODELS)


def _asr_selectors():
    with gr.Row(elem_classes="asr-row"):
        choice = gr.Dropdown(label="Spracherkennungs-Modell", choices=list(ASR_CHOICES), value=ASR_DEFAULT_CHOICE,
                             scale=2, min_width=160)
        language = gr.Dropdown(label="Sprache des Audios", choices=list(ASR_LANGUAGES), value=ASR_DEFAULT_LANGUAGE,
                               scale=1, min_width=140)
    choice.change(_asr_language_visible, inputs=choice, outputs=language, show_progress="hidden")
    return choice, language


def _asr_controls():
    """Modell-/Sprachauswahl + Button, der den Referenztext per Spracherkennung aus dem Referenz-Audio erkennt."""
    choice, language = _asr_selectors()
    button = gr.Button("📝 Referenztext aus Audio erkennen", size="sm")
    return choice, language, button


def asr_info_markdown(choice) -> str:
    key = ASR_CHOICES.get(choice, ASR_CHOICES[ASR_DEFAULT_CHOICE])
    desc, license_ = ASR_DETAILS[key]
    if key in _loaded:
        vram = f"{_model_sizes_gb.get(key, 0):.1f} GB VRAM (geladen)"
    else:
        vram = f"≈ {_vram_estimates.get(key, 0):.1f} GB VRAM"
    lang = " Erkennt die Sprache selbst (keine Sprachauswahl)." if key in AUDIO8_ASR_MODELS else ""
    disk = (f"{_fmt_size(MODEL_DISK_GB[key])} auf der Festplatte" if is_installed(key)
            else f"nicht installiert (Download ≈ {_fmt_size(download_size_gb(key))})")
    return f"{desc}{lang}  \n**Größe:** {disk} · {vram} · **Lizenz:** {license_}"


def _asr_choice_changed(choice):
    key = ASR_CHOICES.get(choice, ASR_CHOICES[ASR_DEFAULT_CHOICE])
    return asr_info_markdown(choice), _asr_language_visible(choice), gr.update(link=MODEL_URLS[key])


QWEN_LONG_TEXT_TIP = ("**Tipp:** Dieses Modell teilt lange Texte nicht automatisch auf – "
                      "für beste Ergebnisse besser kürzere Abschnitte (einige Sätze) generieren.")
SAVE_OUTPUT_HINT = ("Speichert das generierte Audio samt Text als Stimme. Danach in den Tabs unter "
                    "**Voice Cloner** bei „Gespeicherte Stimme“ auswählen und beliebige Texte sprechen lassen.")

TITLE_DESIGN = "Stimme per Beschreibung erstellen"
TITLE_CLONE = "Stimme aus Referenz-Audio klonen"
TITLE_GENERATE = "Text mit vordefinierten Sprechern vorlesen"
TITLE_ASR = "Sprache in Text umwandeln"
TITLE_CLEANUP = "Audio bereinigen"
GITHUB_LINK_TEXT = "🔗 Projekt auf GitHub"


def build_ui():
    _OVERLAYS.clear()
    with gr.Blocks(title="TTS & ASR (lokal)") as demo:
        gr.Markdown(
            f"# TTS & ASR (lokal) <small>v{APP_VERSION}</small>\n"
            + (f"\n> ⚠️ **{UPDATE_HINT}**\n\n" if UPDATE_PENDING else "") +
            "Stimmen per Beschreibung entwerfen (**Voice Designer**), aus Referenz-Audio klonen (**Voice Cloner**), "
            "mit fertigen Sprechern vorlesen (**Voice Generator**), Audio in Text umwandeln (**Speech-to-Text**) "
            "und Aufnahmen von Störgeräuschen befreien (**Audio-Cleanup**). Alle Modelle laufen lokal – fehlende "
            "lassen sich unter „Modelle verwalten“ mit einem Klick installieren.",
            elem_classes="app-header",
        )

        with gr.Accordion("🖥️ GPU & Hintergrund-Aufgaben", open=True):
            gpu_monitor = gr.HTML(gpu_monitor_html())
            with gr.Row(elem_classes="panel-footer"):
                model_msg = _status_box()
                unload_all_btn = gr.Button("⏏ Alle Modelle entladen", variant="stop", size="sm",
                                           scale=0, min_width=220, elem_classes="unload-all")
        with gr.Accordion("📦 Modelle verwalten", open=not any(is_installed(k) for k in MODEL_LABELS)):
            model_table = gr.HTML(model_table_html())
            model_status = gr.Markdown(status_note_markdown(), elem_classes="hint")
            with gr.Accordion("🔑 Hugging-Face-Token (nur für zugangsbeschränkte Modelle)", open=False):
                _hint("Kartoffel-Orpheus ist zugangsbeschränkt: auf "
                      "[huggingface.co](https://huggingface.co/join) anmelden, auf der Modellseite die Bedingungen "
                      "akzeptieren und einen Token vom Typ „Read“ "
                      "([Token erstellen](https://huggingface.co/settings/tokens)) hier eintragen.")
                with gr.Row(elem_classes="voice-row"):
                    hf_token = gr.Textbox(show_label=False, container=False, type="password",
                                          placeholder="hf_...", scale=3)
                    hf_token_btn = gr.Button("💾 Speichern", scale=0, min_width=170, elem_classes="voice-action")
                hf_token_msg = _hint(hf_token_status())
        with gr.Row(elem_classes="hidden-toggles"):
            toggle_buttons = {key: gr.Button(key, elem_id=f"toggle-{key}") for key in MODEL_LABELS
                              if key not in CPU_TOOLS}
            install_buttons = {key: gr.Button(key, elem_id=f"install-{key}") for key in MODEL_LABELS}
            uninstall_buttons = {key: gr.Button(key, elem_id=f"uninstall-{key}") for key in MODEL_LABELS}

        voice_events = []
        with gr.Tabs(elem_classes="cat-tabs"):
            # ---------------------------------------------------------------- Voice Designer
            with gr.Tab("Voice Designer"):
                with gr.Tabs():
                    with gr.Tab(MODEL_LABELS["VoiceDesign"]):
                        _tab_header(
                            TITLE_DESIGN,
                            "Beschreibe Klang, Alter, Emotion und Sprechweise – Qwen3 erzeugt daraus eine neue Stimme. "
                            "10 Sprachen, Beschreibungen auf Englisch wirken am zuverlässigsten. Lizenz: Apache 2.0.",
                            "VoiceDesign",
                        )
                        with _model_area("VoiceDesign"), gr.Row(equal_height=False):
                            with gr.Column(variant="panel"):
                                _section("Eingabe")
                                design_text = _text_input(
                                    "Es ist in der obersten Schublade ... Moment, die ist leer? Das kann nicht sein! "
                                    "Ich bin mir sicher, dass ich es dort hingelegt habe!"
                                )
                                design_language = gr.Dropdown(label="Sprache", choices=LANGUAGES, value="German")
                                design_instruct = gr.Textbox(
                                    label="Stimmbeschreibung", lines=3,
                                    placeholder="Beschreibe die gewünschten Stimmeigenschaften ...",
                                    value="Speak in an incredulous tone, but with a hint of panic beginning to creep "
                                          "into your voice.",
                                )
                                design_btn = _gen_button()
                                _hint(QWEN_LONG_TEXT_TIP)
                            with gr.Column(variant="panel"):
                                _section("Ausgabe")
                                design_audio_out = _output_audio()
                                design_status = _status_box()
                                design_voice_name, design_save_btn = _save_voice_box(SAVE_OUTPUT_HINT)
                        design_last = gr.State(None)

                        design_btn.click(
                            generate_voice_design,
                            inputs=[design_text, design_language, design_instruct],
                            outputs=[design_audio_out, design_status, design_last],
                            show_progress_on=[design_audio_out],
                            **GPU_EVENT,
                        )

                    with gr.Tab(MODEL_LABELS["Parler"]):
                        _tab_header(
                            TITLE_DESIGN,
                            "Gesteuert über eine **englische Beschreibung**: Geschlecht, Tonhöhe (*low/high pitch*), "
                            "Tempo (*slowly/fast*), Ausdruck (*monotone/expressive*), Raumklang (*close up/distant*) und "
                            "Aufnahmequalität (*very clear/noisy audio*). Die Sprache ergibt sich aus dem Text (8 Sprachen "
                            "inkl. Deutsch). Für gleichbleibende deutsche Stimmen einen dieser Sprecher nennen: "
                            f"{', '.join(PARLER_GERMAN_SPEAKERS)}. Läuft in einer eigenen Python-Umgebung "
                            "(`.venv-parler`), lange Texte werden an Satzgrenzen aufgeteilt. Lizenz: Apache 2.0.",
                            "Parler",
                        )
                        with _model_area("Parler"), gr.Row(equal_height=False):
                            with gr.Column(variant="panel"):
                                _section("Eingabe")
                                parler_text = _text_input(
                                    "Hallo und herzlich willkommen! Heute zeige ich dir, wie man mit einer kurzen "
                                    "Beschreibung eine ganz neue Stimme erschafft."
                                )
                                parler_preset = gr.Dropdown(
                                    label="Vorlage", info="Füllt die Stimmbeschreibung aus",
                                    choices=list(PARLER_PRESETS), value=next(iter(PARLER_PRESETS)),
                                )
                                parler_description = gr.Textbox(
                                    label="Stimmbeschreibung", lines=3, info="Auf Englisch",
                                    value=next(iter(PARLER_PRESETS.values())),
                                    placeholder="z. B. A young woman speaks fast with a high pitch. Very clear audio.",
                                )
                                with gr.Accordion("Erweiterte Einstellungen", open=False):
                                    with gr.Row():
                                        parler_seed = _seed()
                                        parler_temperature = _temperature(0.3, 1.5, 1.0)
                                parler_btn = _gen_button()
                            with gr.Column(variant="panel"):
                                _section("Ausgabe")
                                parler_audio_out = _output_audio()
                                parler_status = _status_box()
                                parler_voice_name, parler_save_btn = _save_voice_box(SAVE_OUTPUT_HINT)
                        parler_last = gr.State(None)

                        parler_preset.change(lambda p: PARLER_PRESETS.get(p, gr.update()), inputs=parler_preset,
                                             outputs=parler_description)
                        parler_btn.click(
                            generate_parler,
                            inputs=[parler_text, parler_description, parler_seed, parler_temperature],
                            outputs=[parler_audio_out, parler_status, parler_last],
                            show_progress_on=[parler_audio_out],
                            **GPU_EVENT,
                        )

            # ---------------------------------------------------------------- Voice Cloner
            with gr.Tab("Voice Cloner"):
                with gr.Tabs():
                    with gr.Tab(MODEL_LABELS["Base"]):
                        _tab_header(
                            TITLE_CLONE,
                            "Klont eine Stimme aus 5–15 s Referenz-Audio. Für beste Qualität wird das exakte Transkript "
                            "des Referenz-Audios benötigt – es kann per Spracherkennung erkannt werden. Lizenz: Apache 2.0.",
                            "Base",
                        )
                        with _model_area("Base"), gr.Row(equal_height=False):
                            with gr.Column(variant="panel"):
                                _section("Referenzstimme")
                                clone_voice_select, clone_voice_refresh, clone_voice_delete = _saved_voice_controls()
                                clone_ref_audio = gr.Audio(
                                    label="Referenz-Audio", type="filepath", sources=["upload", "microphone"],
                                )
                                _hint(FORMATS_HINT)
                                clone_asr_model, clone_asr_lang, clone_asr_btn = _asr_controls()
                                clone_ref_text = gr.Textbox(
                                    label="Referenztext (Transkript des Referenz-Audios)", lines=3,
                                    placeholder="Exakt den Text eingeben, der im Referenz-Audio gesprochen wird – "
                                                "oder oben erkennen lassen und danach prüfen ...",
                                )
                                clone_xvector = gr.Checkbox(
                                    label="Nur x-Vector verwenden (kein Referenztext nötig, aber geringere Qualität)",
                                    value=False,
                                )
                                clone_voice_name, clone_save_btn = _save_voice_box(
                                    "Speichert Referenz-Audio und Referenztext als Stimme für alle Klon-Tabs."
                                )
                            with gr.Column():
                                with gr.Column(variant="panel"):
                                    _section("Eingabe")
                                    clone_target_text = _text_input(placeholder="Text für die geklonte Stimme eingeben ...")
                                    clone_language = gr.Dropdown(label="Sprache", choices=LANGUAGES, value="German")
                                    clone_btn = _gen_button()
                                    _hint(QWEN_LONG_TEXT_TIP)
                                with gr.Column(variant="panel"):
                                    _section("Ausgabe")
                                    clone_audio_out = _output_audio()
                                    clone_status = _status_box()

                        clone_btn.click(
                            generate_voice_clone,
                            inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector],
                            outputs=[clone_audio_out, clone_status],
                            show_progress_on=[clone_audio_out],
                            **GPU_EVENT,
                        )
                        clone_asr_btn.click(
                            transcribe_reference, inputs=[clone_ref_audio, clone_asr_model, clone_asr_lang, clone_ref_text],
                            outputs=[clone_ref_text, clone_status], show_progress_on=[clone_ref_text], **GPU_EVENT,
                        )
                        clone_voice_select.select(
                            load_voice, inputs=clone_voice_select,
                            outputs=[clone_ref_audio, clone_ref_text, clone_xvector, clone_language, clone_status],
                        )
                        voice_events.append(clone_voice_refresh.click(lambda: None, outputs=None))
                        voice_events.append(clone_voice_delete.click(
                            delete_voice, inputs=clone_voice_select, outputs=[clone_status, clone_voice_select],
                        ))
                        voice_events.append(clone_save_btn.click(
                            save_clone_reference,
                            inputs=[clone_voice_name, clone_ref_audio, clone_ref_text, clone_language],
                            outputs=[clone_status, clone_voice_select],
                        ))

                    with gr.Tab(MODEL_LABELS["Audio8"]):
                        _tab_header(
                            TITLE_CLONE,
                            "Platz 1 der [TTS Arena](https://tts-agi-tts-arena-v2.hf.space/leaderboard), unterstützt "
                            "Deutsch (die Sprache ergibt sich aus dem Text). **Mit Referenz** wird die Stimme geklont "
                            "(exaktes Transkript nötig), **ohne Referenz** entsteht eine zufällige Stimme – der Seed legt "
                            "sie fest. Lange Texte werden aufgeteilt, die Stimme bleibt gleich. "
                            "Lizenz: Apache 2.0 (Preview).",
                            "Audio8",
                        )
                        with _model_area("Audio8"), gr.Row(equal_height=False):
                            with gr.Column(variant="panel"):
                                _section("Referenzstimme (optional)")
                                a8_voice_select, a8_voice_refresh, a8_voice_delete = _saved_voice_controls()
                                a8_ref_audio = gr.Audio(
                                    label="Referenz-Audio", type="filepath", sources=["upload", "microphone"],
                                )
                                _hint(FORMATS_HINT + " Höchstens 30 s. Leer lassen für eine zufällige Stimme.")
                                a8_asr_model, a8_asr_lang, a8_asr_btn = _asr_controls()
                                a8_ref_text = gr.Textbox(
                                    label="Referenztext (Transkript des Referenz-Audios)", lines=3,
                                    placeholder="Text eingeben oder oben erkennen lassen und danach prüfen ...",
                                )
                            with gr.Column():
                                with gr.Column(variant="panel"):
                                    _section("Eingabe")
                                    a8_text = _text_input(
                                        "Hallo und herzlich willkommen! Schön, dass du da bist. Heute erzähle ich dir "
                                        "eine kleine Geschichte über einen mutigen Gnom."
                                    )
                                    with gr.Accordion("Erweiterte Einstellungen", open=False):
                                        with gr.Row():
                                            a8_seed = _seed("Ohne Referenz bestimmt der Seed die Stimme")
                                            a8_temperature = _temperature(0.3, 1.2, 0.7)
                                        with gr.Row():
                                            a8_top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.01, label="Top-p",
                                                                 info=TOP_P_INFO)
                                            a8_top_k = gr.Slider(1, 200, value=50, step=1, label="Top-k",
                                                                 info="Höchstens so viele Möglichkeiten pro Schritt")
                                    a8_btn = _gen_button()
                                with gr.Column(variant="panel"):
                                    _section("Ausgabe")
                                    a8_audio_out = _output_audio()
                                    a8_status = _status_box()
                                    a8_voice_name, a8_save_btn = _save_voice_box(SAVE_OUTPUT_HINT)
                        a8_last = gr.State(None)

                        a8_btn.click(
                            generate_audio8,
                            inputs=[a8_text, a8_ref_audio, a8_ref_text, a8_temperature, a8_top_p, a8_top_k, a8_seed],
                            outputs=[a8_audio_out, a8_status, a8_last],
                            show_progress_on=[a8_audio_out],
                            **GPU_EVENT,
                        )
                        a8_asr_btn.click(
                            transcribe_reference, inputs=[a8_ref_audio, a8_asr_model, a8_asr_lang, a8_ref_text],
                            outputs=[a8_ref_text, a8_status], show_progress_on=[a8_ref_text], **GPU_EVENT,
                        )
                        a8_voice_select.select(
                            load_voice_reference, inputs=a8_voice_select, outputs=[a8_ref_audio, a8_ref_text, a8_status],
                        )
                        voice_events.append(a8_voice_refresh.click(lambda: None, outputs=None))
                        voice_events.append(a8_voice_delete.click(
                            delete_voice, inputs=a8_voice_select, outputs=[a8_status, a8_voice_select],
                        ))
                        voice_events.append(a8_save_btn.click(
                            save_audio8_voice,
                            inputs=[a8_voice_name, a8_audio_out, a8_last],
                            outputs=[a8_status, clone_voice_select],
                        ))

                    with gr.Tab(MODEL_LABELS["Chatterbox"]):
                        _tab_header(
                            TITLE_CLONE,
                            "Alternative zum Qwen-Klon: **kein Transkript nötig**, 5–15 s Referenz-Audio genügen. Ohne "
                            "Referenz spricht die eingebaute (englische) Standardstimme. Ist die Referenz nicht auf Deutsch, "
                            "CFG auf 0 setzen, damit kein Akzent übernommen wird. Läuft in einer eigenen Python-Umgebung "
                            "(`.venv-chatterbox`), lange Texte werden aufgeteilt. Ausgaben enthalten ein unhörbares "
                            "Wasserzeichen (Perth). Lizenz: MIT.",
                            "Chatterbox",
                        )
                        with _model_area("Chatterbox"), gr.Row(equal_height=False):
                            with gr.Column(variant="panel"):
                                _section("Referenzstimme (optional)")
                                cb_voice_select, cb_voice_refresh, cb_voice_delete = _saved_voice_controls(
                                    "Füllt das Referenz-Audio automatisch aus"
                                )
                                cb_ref_audio = gr.Audio(
                                    label="Referenz-Audio", type="filepath", sources=["upload", "microphone"],
                                )
                                _hint(FORMATS_HINT + " Leer lassen für die Standardstimme.")
                            with gr.Column():
                                with gr.Column(variant="panel"):
                                    _section("Eingabe")
                                    cb_text = _text_input(
                                        "Dann hoffen wir mal, dass die Gilde eine brauchbare Einführung für verlorene "
                                        "Fremde hat."
                                    )
                                    with gr.Row():
                                        cb_exaggeration = gr.Slider(
                                            0.25, 2.0, value=0.5, step=0.05, label="Ausdrucksstärke",
                                            info="0,5 = neutral, höher = emotionaler (und schneller); ausdrucksstark: ~0,7",
                                        )
                                        cb_cfg = gr.Slider(
                                            0.0, 1.0, value=0.5, step=0.05, label="CFG / Tempo",
                                            info="Niedriger = langsamer, freier; ausdrucksstark: ~0,3",
                                        )
                                    with gr.Accordion("Erweiterte Einstellungen", open=False):
                                        with gr.Row():
                                            cb_seed = _seed()
                                            cb_temperature = _temperature(0.05, 2.0, 0.8)
                                    cb_btn = _gen_button()
                                with gr.Column(variant="panel"):
                                    _section("Ausgabe")
                                    cb_audio_out = _output_audio()
                                    cb_status = _status_box()

                        cb_btn.click(
                            generate_chatterbox,
                            inputs=[cb_text, cb_ref_audio, cb_exaggeration, cb_cfg, cb_temperature, cb_seed],
                            outputs=[cb_audio_out, cb_status],
                            show_progress_on=[cb_audio_out],
                            **GPU_EVENT,
                        )
                        cb_voice_select.select(
                            lambda name: (lambda r: (r[0], r[2]))(load_voice_reference(name)),
                            inputs=cb_voice_select, outputs=[cb_ref_audio, cb_status],
                        )
                        voice_events.append(cb_voice_refresh.click(lambda: None, outputs=None))
                        voice_events.append(cb_voice_delete.click(
                            delete_voice, inputs=cb_voice_select, outputs=[cb_status, cb_voice_select],
                        ))

            # ---------------------------------------------------------------- Voice Generator
            with gr.Tab("Voice Generator"):
                with gr.Tabs():
                    with gr.Tab(MODEL_LABELS["CustomVoice"]):
                        _tab_header(
                            TITLE_GENERATE,
                            "9 vordefinierte Qwen3-Sprecher in 10 Sprachen. Die Stilanweisung beeinflusst Emotion, "
                            "Tempo und Betonung. Lizenz: Apache 2.0.",
                            "CustomVoice",
                        )
                        with _model_area("CustomVoice"), gr.Row(equal_height=False):
                            with gr.Column(variant="panel"):
                                _section("Eingabe")
                                tts_text = _text_input(
                                    "Hallo! Willkommen beim Text-to-Speech-System. Dies ist eine Demo unserer "
                                    "Sprachsynthese."
                                )
                                with gr.Row():
                                    tts_language = gr.Dropdown(label="Sprache", choices=LANGUAGES, value="German")
                                    tts_speaker = gr.Dropdown(label="Sprecher", choices=SPEAKERS, value="Ryan")
                                tts_instruct = gr.Textbox(
                                    label="Stilanweisung (optional)", lines=2,
                                    placeholder="z. B. Very happy. / Speak angrily. / Speak slowly and sadly.",
                                    info=(
                                        "Wirkt bei diesem Modell nur subtil – die Sprecherstimme bleibt erhalten. Für "
                                        "deutliches Flüstern, Schreien o. Ä. den Voice Designer nutzen. Englische "
                                        "Anweisungen wirken am zuverlässigsten."
                                    ),
                                )
                                tts_btn = _gen_button()
                                _hint(QWEN_LONG_TEXT_TIP)
                            with gr.Column(variant="panel"):
                                _section("Ausgabe")
                                tts_audio_out = _output_audio()
                                tts_status = _status_box()

                        tts_btn.click(
                            generate_custom_voice,
                            inputs=[tts_text, tts_language, tts_speaker, tts_instruct],
                            outputs=[tts_audio_out, tts_status],
                            show_progress_on=[tts_audio_out],
                            **GPU_EVENT,
                        )

                    with gr.Tab("Kartoffel-Orpheus"):
                        _tab_header(
                            TITLE_GENERATE,
                            "Speziell auf Deutsch trainiert. **Natural**: 19 natürliche Sprecher (wenig Emotion). "
                            "**Synthetic**: 4 Sprecher mit wählbarer Emotion; Ausrufe wie "
                            + ", ".join(f"„{o}“" for o in kartoffel_tts.OUTBURSTS)
                            + " können direkt im Text stehen. Lange Texte werden an Satzgrenzen aufgeteilt; ca. 2–3 s "
                            "Rechenzeit pro Sekunde Audio. Lizenz: Llama 3.2 Community License.",
                            "KartoffelNatural", "KartoffelSynthetic",
                            texts=("🤗 Natural auf Hugging Face", "🤗 Synthetic auf Hugging Face"),
                        )
                        kart_variant = gr.Radio(
                            label="Variante", choices=list(KARTOFFEL_VARIANTS),
                            value=next(iter(KARTOFFEL_VARIANTS)),
                        )
                        with _model_area(lambda v: KARTOFFEL_VARIANTS.get(v, "KartoffelNatural"), kart_variant), \
                                gr.Row(equal_height=False):
                            with gr.Column(variant="panel"):
                                _section("Eingabe")
                                kart_text = _text_input(
                                    "Tief im verwunschenen Wald, wo die Bäume uralte Geheimnisse flüsterten, lebte ein "
                                    "kleiner Gnom namens Fips, der die Sprache der Tiere verstand.",
                                    placeholder="Deutschen Text eingeben ...",
                                )
                                with gr.Row():
                                    kart_speaker = gr.Dropdown(
                                        label="Sprecher", choices=kartoffel_tts.NATURAL_SPEAKERS,
                                        value=kartoffel_tts.NATURAL_SPEAKERS[0], allow_custom_value=True,
                                    )
                                    kart_emotion = gr.Dropdown(
                                        label="Emotion", choices=kartoffel_tts.EMOTIONS, value="Neutral", visible=False,
                                    )
                                with gr.Accordion("Erweiterte Einstellungen", open=False):
                                    with gr.Row():
                                        kart_temperature = _temperature(0.1, 1.5, 0.6)
                                        kart_top_p = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="Top-p",
                                                               info=TOP_P_INFO)
                                    kart_rep = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition Penalty",
                                                         info="Mindestens 1.1 empfohlen, sonst kann das Modell hängen bleiben")
                                kart_btn = _gen_button()
                            with gr.Column(variant="panel"):
                                _section("Ausgabe")
                                kart_audio_out = _output_audio()
                                kart_status = _status_box()

                        kart_variant.change(_kartoffel_variant_changed, inputs=kart_variant,
                                            outputs=[kart_speaker, kart_emotion])
                        kart_btn.click(
                            generate_kartoffel,
                            inputs=[kart_text, kart_variant, kart_speaker, kart_emotion, kart_temperature, kart_top_p,
                                    kart_rep],
                            outputs=[kart_audio_out, kart_status],
                            show_progress_on=[kart_audio_out],
                            **GPU_EVENT,
                        )

            # ---------------------------------------------------------------- Speech-to-Text
            with gr.Tab("Speech-to-Text"):
                (asr_hf,) = _tab_header(
                    TITLE_ASR,
                    "Lange Aufnahmen werden automatisch an leisen Stellen in Abschnitte von höchstens 28 s geteilt. "
                    "Dieselbe Erkennung gibt es auch in den Klon-Tabs, um den Referenztext auszufüllen.",
                    ASR_CHOICES[ASR_DEFAULT_CHOICE],
                )
                with gr.Column(variant="panel"):
                    asr_choice, asr_language = _asr_selectors()
                    asr_info = gr.Markdown(asr_info_markdown(ASR_DEFAULT_CHOICE), elem_classes="hint model-info")
                with _model_area(lambda v: ASR_CHOICES.get(v, ASR_CHOICES[ASR_DEFAULT_CHOICE]), asr_choice), \
                        gr.Row(equal_height=False):
                    with gr.Column(variant="panel"):
                        _section("Eingabe")
                        asr_audio = _input_audio()
                        _hint(FORMATS_HINT.split(" Empfehlung:")[0])
                        asr_btn = _gen_button("📝 Text erkennen")
                    with gr.Column(variant="panel"):
                        _section("Ausgabe")
                        asr_text = gr.Textbox(
                            label="Erkannter Text", lines=10, interactive=True, buttons=["copy"],
                            placeholder="Hier erscheint der erkannte Text ...",
                        )
                        asr_status = _status_box()

                asr_choice.change(_asr_choice_changed, inputs=asr_choice, outputs=[asr_info, asr_language, asr_hf],
                                  show_progress="hidden")
                asr_btn.click(
                    transcribe_audio, inputs=[asr_audio, asr_choice, asr_language], outputs=[asr_text, asr_status],
                    show_progress_on=[asr_text], **GPU_EVENT,
                ).then(asr_info_markdown, inputs=asr_choice, outputs=asr_info, show_progress="hidden")

            # ---------------------------------------------------------------- Audio-Cleanup
            with gr.Tab("Audio-Cleanup"):
                with gr.Tabs():
                    with gr.Tab(MODEL_LABELS["Clear"]):
                        _tab_header(
                            TITLE_CLEANUP,
                            "Entfernt Rauschen, Brummen und Hintergrundgeräusche aus Sprachaufnahmen – sehr schnell, "
                            "läuft auf dem Prozessor (kein VRAM). **Studio** filtert maximal, **Natural** klingt "
                            "natürlicher. Lizenz: source-available (Bedingungen auf der Modellseite).",
                            "Clear",
                        )
                        with _model_area("Clear"), gr.Row(equal_height=False):
                            with gr.Column(variant="panel"):
                                _section("Eingabe")
                                clear_in = _input_audio()
                                _hint(CLEANUP_FORMATS_HINT)
                                clear_variant = gr.Radio(label="Variante", choices=list(CLEAR_VARIANT_CHOICES),
                                                         value=next(iter(CLEAR_VARIANT_CHOICES)))
                                clear_strength = gr.Slider(0, 100, value=100, step=5, label="Stärke (%)",
                                                           info="100 = volle Wirkung; niedriger = Störgeräusche "
                                                                "nur abschwächen")
                                clear_norm = gr.Checkbox(label="Lautstärke normalisieren", value=True)
                                clear_btn = _gen_button("🧹 Bereinigen")
                            with gr.Column(variant="panel"):
                                _section("Ausgabe")
                                clear_out = _output_audio("Bereinigtes Audio")
                                clear_status = _status_box()
                        clear_btn.click(
                            clean_clear, inputs=[clear_in, clear_variant, clear_strength, clear_norm],
                            outputs=[clear_out, clear_status], show_progress_on=[clear_out],
                            concurrency_id="cpu-cleanup", concurrency_limit=1,
                        )

                    with gr.Tab(MODEL_LABELS["ClearVoice"]):
                        _tab_header(
                            TITLE_CLEANUP,
                            "Werkzeugkasten von Alibaba: **Rauschunterdrückung** (drei Modelle), **Sprecher trennen** "
                            "(zwei gleichzeitig sprechende Personen in zwei Spuren) und **Qualität erhöhen** (dumpfe "
                            "Aufnahmen auf 48 kHz hochrechnen). Läuft in einer eigenen Python-Umgebung "
                            "(`.venv-clearvoice`) auf der GPU. Lizenz: Apache 2.0.",
                            "ClearVoice", texts=(GITHUB_LINK_TEXT,),
                        )
                        with _model_area("ClearVoice"), gr.Row(equal_height=False):
                            with gr.Column(variant="panel"):
                                _section("Eingabe")
                                cv_in = _input_audio()
                                _hint(FORMATS_HINT.split(" Empfehlung:")[0])
                                cv_model = gr.Dropdown(label="Aufgabe / Modell", choices=list(CLEARVOICE_MODELS),
                                                       value=next(iter(CLEARVOICE_MODELS)))
                                cv_info = _hint(next(iter(CLEARVOICE_MODELS.values()))[1])
                                cv_norm = gr.Checkbox(label="Lautstärke normalisieren", value=True)
                                cv_btn = _gen_button("🧹 Verarbeiten")
                            with gr.Column(variant="panel"):
                                _section("Ausgabe")
                                cv_out = _output_audio("Bereinigtes Audio")
                                cv_out2 = gr.Audio(label="Sprecher 2", type="numpy", visible=False)
                                cv_status = _status_box()
                        cv_model.change(_clearvoice_choice_changed, inputs=cv_model, outputs=[cv_info, cv_out],
                                        show_progress="hidden")
                        cv_btn.click(
                            clean_clearvoice, inputs=[cv_in, cv_model, cv_norm], outputs=[cv_out, cv_out2, cv_status],
                            show_progress_on=[cv_out], **GPU_EVENT,
                        )

                    with gr.Tab(MODEL_LABELS["DeepFilterNet"]):
                        _tab_header(
                            TITLE_CLEANUP,
                            "Bewährte, sehr gründliche Rauschunterdrückung für Sprache (48 kHz). Entfernt mehr "
                            "Störgeräusche als Clear, braucht dafür etwas länger. Läuft auf dem Prozessor (kein VRAM). "
                            "Lizenz: MIT / Apache 2.0.",
                            "DeepFilterNet", texts=(GITHUB_LINK_TEXT,),
                        )
                        with _model_area("DeepFilterNet"), gr.Row(equal_height=False):
                            with gr.Column(variant="panel"):
                                _section("Eingabe")
                                dfn_in = _input_audio()
                                _hint(CLEANUP_FORMATS_HINT)
                                dfn_atten = gr.Slider(6, 100, value=100, step=1, label="Maximale Absenkung (dB)",
                                                      info="100 = unbegrenzt; z. B. 20–30 lässt etwas Umgebung "
                                                           "hörbar und klingt natürlicher")
                                dfn_pf = gr.Checkbox(label="Nachfilter", value=False,
                                                     info="Entfernt Restrauschen stärker, kann Sprache leicht dämpfen")
                                dfn_norm = gr.Checkbox(label="Lautstärke normalisieren", value=True)
                                dfn_btn = _gen_button("🧹 Bereinigen")
                            with gr.Column(variant="panel"):
                                _section("Ausgabe")
                                dfn_out = _output_audio("Bereinigtes Audio")
                                dfn_status = _status_box()
                        dfn_btn.click(
                            clean_deepfilter, inputs=[dfn_in, dfn_atten, dfn_pf, dfn_norm],
                            outputs=[dfn_out, dfn_status], show_progress_on=[dfn_out],
                            concurrency_id="cpu-cleanup", concurrency_limit=1,
                        )

        # Speichern-Buttons aktualisieren die Stimmenliste im Tab Qwen3 Voice Clone.
        voice_events.append(design_save_btn.click(
            save_design_voice,
            inputs=[design_voice_name, design_audio_out, design_last],
            outputs=[design_status, clone_voice_select],
        ))
        voice_events.append(parler_save_btn.click(
            save_parler_voice,
            inputs=[parler_voice_name, parler_audio_out, parler_last],
            outputs=[parler_status, clone_voice_select],
        ))

        # Neue/gelöschte Stimmen in allen Auswahllisten anzeigen (Auswahl bleibt erhalten).
        voice_dropdowns = [clone_voice_select, a8_voice_select, cb_voice_select]
        for event in voice_events:
            event.then(lambda: [_voice_choices_update()] * len(voice_dropdowns), outputs=voice_dropdowns,
                       show_progress="hidden")

        panel_outputs = [model_table, model_status]
        for key, btn in toggle_buttons.items():
            btn.click(
                _make_toggle_handler(key), outputs=[model_msg, *panel_outputs],
                show_progress="hidden", **GPU_EVENT,
            )
        for key, btn in install_buttons.items():
            btn.click(_make_install_handler(key), outputs=[model_msg, *panel_outputs], show_progress="hidden")
        for key, btn in uninstall_buttons.items():
            btn.click(_make_uninstall_handler(key), outputs=[model_msg, *panel_outputs], show_progress="hidden")
        unload_all_btn.click(
            _unload_all_handler, outputs=[model_msg, *panel_outputs],
            show_progress="hidden", **GPU_EVENT,
        )
        hf_token_btn.click(save_hf_token, inputs=hf_token, outputs=[hf_token, hf_token_msg])

        # Installations-Overlays: Sichtbarkeit/Fortschritt je nach Modell (ggf. abhängig von einer Auswahl).
        sources = []
        for entry in _OVERLAYS:
            if entry["source"] is not None and all(entry["source"] is not s for s in sources):
                sources.append(entry["source"])

        def _source_value(entry, values):
            if entry["source"] is None:
                return None
            return values[next(i for i, s in enumerate(sources) if s is entry["source"])]

        overlay_outputs = [c for e in _OVERLAYS for c in (e["col"], e["md"], e["btn"])]

        def overlay_updates(*values):
            out = []
            for entry in _OVERLAYS:
                out += _overlay_state(_overlay_key(entry, _source_value(entry, values)))
            return out

        for entry in _OVERLAYS:
            inputs = [entry["source"]] if entry["source"] is not None else []
            entry["btn"].click(
                lambda *v, e=entry: request_install(_overlay_key(e, v[0] if v else None)),
                inputs=inputs, outputs=model_msg, show_progress="hidden",
            ).then(overlay_updates, inputs=sources, outputs=overlay_outputs, show_progress="hidden")
        for source in sources:
            source.change(overlay_updates, inputs=sources, outputs=overlay_outputs, show_progress="hidden")

        status_timer = gr.Timer(1)
        status_timer.tick(_model_panel_updates, outputs=panel_outputs, show_progress="hidden", concurrency_limit=None)
        status_timer.tick(overlay_updates, inputs=sources, outputs=overlay_outputs, show_progress="hidden",
                          concurrency_limit=None)
        gpu_timer = gr.Timer(0.5)
        gpu_timer.tick(gpu_monitor_html, outputs=gpu_monitor, show_progress="hidden", concurrency_limit=None)
        demo.load(_model_panel_updates, outputs=panel_outputs, show_progress="hidden", concurrency_limit=None)
        demo.load(overlay_updates, inputs=sources, outputs=overlay_outputs, show_progress="hidden",
                  concurrency_limit=None)
        demo.load(lambda: [_voice_choices_update()] * len(voice_dropdowns), outputs=voice_dropdowns,
                  show_progress="hidden", concurrency_limit=None)

        gr.Markdown(
            "---\nLokal ausgeführte Modelle: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Alibaba Qwen) · "
            "[Kartoffel-Orpheus](https://huggingface.co/SebastianBodza) · "
            "[Parler-TTS](https://github.com/huggingface/parler-tts) (Hugging Face) · "
            "[Audio8-TTS & Audio8-ASR](https://huggingface.co/Edge0) (Edge0) · "
            "[Chatterbox](https://github.com/resemble-ai/chatterbox) (Resemble AI) · "
            "[Whisper Turbo German](https://huggingface.co/primeline/whisper-large-v3-turbo-german) (primeline) · "
            "[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (Alibaba Qwen) · "
            "[CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper) (nyra health) · "
            "[Clear](https://huggingface.co/desert-ant-labs/clear) (Desert Ant Labs) · "
            "[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) (Alibaba Tongyi Lab) · "
            "[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) (Hendrik Schröter)",
            elem_classes="hint",
        )

    return demo

CONSOLE_HELP = """
Commands (type in this window and press Enter):
  help                       Show this help
  status                     Show models (size, state) and GPU memory
  load <model>               Load a model
                             (model: design | clone | tts | natural | synthetic | parler | audio8 | chatterbox
                                     | whisper | qwen-asr | crisper | asr-high | asr-medium | asr-low
                                     | clearvoice)
  unload <model|all>         Unload model(s)
  install <model>            Download & install a missing model (also: clear | deepfilter)
  uninstall <model>          Delete a model's files to free disk space (can be reinstalled later)
  update                     Check Hugging Face for new model versions and download them
  open                       Open the web interface in the browser
  exit                       Unload all models and quit
Closing this window also unloads all models.
"""


def _shutdown(demo) -> None:
    print("Entlade alle Modelle ...", flush=True)
    unload_all(timeout=10)
    with contextlib.suppress(Exception):
        demo.close()
    print("Beendet.", flush=True)
    os._exit(0)


def _install_close_handler() -> None:
    """Entlädt die Modelle, wenn das Konsolenfenster geschlossen wird (Windows)."""
    if os.name != "nt":
        return
    import ctypes

    ctrl_close, ctrl_logoff, ctrl_shutdown = 2, 5, 6
    handler_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)

    def handler(ctrl_type):
        if ctrl_type in (ctrl_close, ctrl_logoff, ctrl_shutdown):
            unload_all(timeout=3)
        return 0

    _install_close_handler.ref = handler_type(handler)
    ctypes.windll.kernel32.SetConsoleCtrlHandler(_install_close_handler.ref, True)


def _alias_for(key: str) -> str:
    return next((a for a, k in MODEL_ALIASES.items() if k == key), key)


def _print_status() -> None:
    for key, label in MODEL_LABELS.items():
        if is_installed(key):
            print(f"  {label:<28} {_fmt_size(MODEL_DISK_GB[key]):>7} auf Festplatte · {_model_state(key)}")
        else:
            print(f"  {label:<28} {_model_state(key)}  ->  install {_alias_for(key)}")
    print(f"  {gpu_summary()}")


def _console_loop(demo) -> None:
    import webbrowser

    print(CONSOLE_HELP, flush=True)
    models = ("design | clone | tts | natural | synthetic | parler | audio8 | chatterbox"
              " | whisper | qwen-asr | crisper | asr-high | asr-medium | asr-low | clearvoice")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            # Keine interaktive Konsole: Server einfach weiterlaufen lassen.
            demo.block_thread()
            return
        except KeyboardInterrupt:
            _shutdown(demo)
        if not line:
            continue
        cmd, _, arg = line.partition(" ")
        cmd, arg = cmd.lower(), arg.strip().lower()

        if cmd in ("help", "?"):
            print(CONSOLE_HELP)
        elif cmd == "status":
            _print_status()
        elif cmd in ("exit", "quit"):
            _shutdown(demo)
        elif cmd == "open":
            webbrowser.open(URL)
        elif cmd == "update":
            start_model_updates()
        elif cmd == "load":
            key = MODEL_ALIASES.get(arg)
            if key is None:
                print(f"Unbekanntes Modell. Erlaubt: load {models}")
                continue
            try:
                get_model(key)
            except Exception as e:  # noqa: BLE001
                print(f"Fehler: {type(e).__name__}: {e}")
        elif cmd == "install":
            key = MODEL_ALIASES.get(arg)
            if key is None:
                print(f"Unbekanntes Modell. Erlaubt: install {models} | clear | deepfilter")
                continue
            print(request_install(key))
        elif cmd == "uninstall":
            key = MODEL_ALIASES.get(arg)
            if key is None:
                print(f"Unbekanntes Modell. Erlaubt: uninstall {models} | clear | deepfilter")
                continue
            answer = input(f"{MODEL_LABELS[key]} wirklich deinstallieren (Dateien löschen)? [y/N] ").strip().lower()
            if answer[:1] in ("y", "j"):
                print(uninstall_model(key))
        elif cmd == "unload":
            if arg in ("all", ""):
                if not unload_all():
                    print("Es war kein Modell geladen.")
            elif arg in MODEL_ALIASES:
                if not unload_model(MODEL_ALIASES[arg]):
                    print(f"{MODEL_LABELS[MODEL_ALIASES[arg]]} war nicht geladen.")
            else:
                print(f"Unbekanntes Modell. Erlaubt: unload {models} | all")
        else:
            print(f"Unbekannter Befehl: {cmd!r} – 'help' zeigt alle Befehle.")


if __name__ == "__main__":
    _step("Baue Weboberfläche auf ...")
    VOICES_DIR.mkdir(exist_ok=True)
    theme = gr.themes.Soft(font=["Source Sans Pro", "Arial", "sans-serif"])
    css = ".gradio-container {max-width: none !important;}\n" + GPU_CSS + LAYOUT_CSS
    demo = build_ui()
    open_browser = os.environ.get("QWEN_NO_BROWSER") != "1"
    _step(f"Starte Webserver auf {URL} ...")
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=PORT,
        inbrowser=open_browser,
        theme=theme,
        css=css,
        js=TOGGLE_JS,
        prevent_thread_lock=True,
        quiet=True,
        allowed_paths=[str(VOICES_DIR)],
    )
    _install_close_handler()
    _step(f"Bereit! Weboberfläche: {URL}" + (" (wird im Browser geöffnet)" if open_browser else ""))
    print("Modelle werden erst bei Bedarf geladen (oder per Button / 'load <model>').")
    print("Dieses Fenster offen lassen, solange die Weboberfläche genutzt wird.")
    if UPDATE_PENDING:
        print("!" * 70)
        print(f"! {UPDATE_HINT}")
        print("!" * 70)
    _print_status()
    if AUTO_UPDATE:
        start_model_updates()
    else:
        _update_summary = "Automatische Updates deaktiviert (QWEN_TTS_AUTO_UPDATE=0) – 'update' in der Konsole."
    _console_loop(demo)