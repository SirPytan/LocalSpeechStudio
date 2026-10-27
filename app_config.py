"""Gemeinsame Einstellungen: Modell-Ordner, Torch-Backend und Pfade der Python-Umgebungen.

Die Werte stehen in config.json im Programmordner (wird vom Installer angelegt):
    {"models_dir": "models", "torch_backend": "cuda"}
Relative Pfade beziehen sich auf den Programmordner. Die Umgebungsvariable QTTS_MODELS_DIR hat Vorrang.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.json"
IS_WINDOWS = os.name == "nt"

# Zusätzliche Python-Umgebungen für Modelle mit abweichenden Bibliotheksversionen.
SIDE_ENVS = {
    "parler": ".venv-parler",
    "chatterbox": ".venv-chatterbox",
    "clearvoice": ".venv-clearvoice",
}


def load_config() -> dict:
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(**values) -> dict:
    data = load_config()
    data.update(values)
    CONFIG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def resolve_models_dir(value: str | os.PathLike | None = None) -> Path:
    raw = value or os.environ.get("QTTS_MODELS_DIR") or load_config().get("models_dir") or "models"
    path = Path(os.path.expandvars(os.path.expanduser(str(raw))))
    return (path if path.is_absolute() else BASE_DIR / path).resolve()


MODELS_DIR = resolve_models_dir()


def venv_python(env_dir: str | Path) -> Path:
    env = Path(env_dir)
    env = env if env.is_absolute() else BASE_DIR / env
    return env / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def side_env_python(name: str) -> Path:
    return venv_python(SIDE_ENVS[name])


def torch_backend() -> str:
    """cuda | rocm | cpu – aus config.json, sonst aus der installierten torch-Version abgeleitet."""
    backend = load_config().get("torch_backend")
    if backend in ("cuda", "rocm", "cpu"):
        return backend
    try:
        import torch

        if getattr(torch.version, "hip", None):
            return "rocm"
        if torch.version.cuda:
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"
