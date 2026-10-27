"""Basis für Modelle, die in einer eigenen Python-Umgebung als Hintergrundprozess laufen.

Manche Modelle benötigen andere Bibliotheksversionen (z. B. transformers) als Qwen3-TTS und laufen
deshalb in einer eigenen venv. Die Kommunikation erfolgt über JSON-Zeilen auf stdin/stdout, das
Audio wird als .npy-Datei übergeben. "Entladen" beendet den Prozess, womit der VRAM vollständig frei wird.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

from app_config import BASE_DIR, MODELS_DIR, side_env_python  # noqa: F401


class WorkerClient:
    NAME = "Worker"
    SCRIPT: Path
    LOG_FILE: Path
    SETUP_HINT = "Bitte in der Modell-Übersicht auf „Installieren“ klicken."
    ENV = ""  # Name der Umgebung in app_config.SIDE_ENVS
    CWD: Path | None = None

    @classmethod
    def python(cls) -> Path:
        return side_env_python(cls.ENV)

    def __init__(self, proc: subprocess.Popen, ready: dict):
        self._proc = proc
        self._lock = threading.Lock()
        self._next_id = 0
        self.sampling_rate = int(ready["sampling_rate"])
        self.vram_gb = float(ready.get("vram_gb", 0.0))
        self.device = ready.get("device", "?")
        self.last_sr = self.sampling_rate

    @classmethod
    def _log_tail(cls, lines: int = 12) -> str:
        try:
            text = cls.LOG_FILE.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            return "\n".join(text[-lines:])
        except OSError:
            return ""

    @classmethod
    def start(cls, *args) -> "WorkerClient":
        python = cls.python()
        if not python.exists():
            raise FileNotFoundError(f"{cls.NAME}-Umgebung fehlt ({python}). {cls.SETUP_HINT}")
        cwd = cls.CWD or BASE_DIR
        cwd.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                   HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", TQDM_DISABLE="1")
        log = cls.LOG_FILE.open("w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            [str(python), str(cls.SCRIPT), *map(str, args)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env, cwd=str(cwd),
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        log.close()
        try:
            msg = cls._read_message(proc)
        except Exception:
            proc.kill()
            raise
        if msg.get("event") != "ready":
            proc.kill()
            raise RuntimeError(f"{cls.NAME}-Worker konnte nicht starten: {msg.get('message', msg)}")
        return cls(proc, msg)

    @classmethod
    def _read_message(cls, proc: subprocess.Popen) -> dict:
        while True:
            line = proc.stdout.readline()
            if not line:
                proc.wait(timeout=5)
                raise RuntimeError(
                    f"{cls.NAME}-Worker wurde unerwartet beendet (Code {proc.returncode}).\n{cls._log_tail()}"
                )
            line = line.strip()
            if line.startswith("{"):
                msg = json.loads(line)
                if "event" in msg:
                    return msg

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def request(self, **params) -> np.ndarray:
        """Sendet einen generate-Auftrag und liefert das Audio als float32-Array."""
        with self._lock:
            if not self.alive:
                raise RuntimeError(f"{self.NAME}-Worker läuft nicht mehr – Modell bitte neu laden.")
            self._next_id += 1
            fd, out = tempfile.mkstemp(suffix=".npy", prefix=f"{self.NAME.lower()}_")
            os.close(fd)
            try:
                req = {"cmd": "generate", "id": self._next_id, "out": out, **params}
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
                msg = self._read_message(self._proc)
                if msg.get("event") != "done":
                    raise RuntimeError(msg.get("message", str(msg)))
                self.last_sr = int(msg.get("sr") or self.sampling_rate)
                if msg.get("vram_gb") is not None:
                    self.vram_gb = max(self.vram_gb, float(msg["vram_gb"]))
                return np.load(out)
            finally:
                Path(out).unlink(missing_ok=True)

    def close(self) -> None:
        """Beendet den Worker; der VRAM wird dabei vom Treiber freigegeben."""
        if self.alive:
            try:  # erst sauber beenden (venv-Launcher + eigentlicher Python-Prozess), dann hart
                self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                self._proc.kill()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def worker_main(load, generate) -> None:
    """Gemeinsame Hauptschleife der Worker-Skripte.

    load() -> (state, ready_dict); generate(state, request) -> np.ndarray (float32)
    oder (np.ndarray, Abtastrate), falls die Rate vom Auftrag abhängt.
    """
    import sys

    proto = sys.stdout
    sys.stdout = sys.stderr  # Ausgaben von Bibliotheken nicht ins Protokoll mischen

    def send(**msg) -> None:
        proto.write(json.dumps(msg) + "\n")
        proto.flush()

    try:
        state, ready = load()
    except Exception as e:  # noqa: BLE001
        send(event="fatal", message=f"{type(e).__name__}: {e}")
        raise
    send(event="ready", **ready)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("cmd") == "quit":
            break
        try:
            result = generate(state, req)
            sr = None
            if isinstance(result, tuple):
                result, sr = result
            wav = np.asarray(result, dtype=np.float32).squeeze()
            np.save(req["out"], wav)
            vram = None
            torch_mod = sys.modules.get("torch")
            if torch_mod is not None and torch_mod.cuda.is_available():
                vram = torch_mod.cuda.memory_reserved() / 1024**3
            send(event="done", id=req.get("id"), samples=int(wav.shape[-1]), sr=sr, vram_gb=vram)
        except Exception as e:  # noqa: BLE001
            send(event="error", id=req.get("id"), message=f"{type(e).__name__}: {e}")
