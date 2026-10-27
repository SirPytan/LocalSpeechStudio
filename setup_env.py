"""Richtet die Python-Umgebungen der App ein (wird von install.bat / install.sh und von der App genutzt).

    python setup_env.py --env main                  Hauptumgebung (.venv) inkl. Einstellungen
    python setup_env.py --env clearvoice            Zusatzumgebung (parler | chatterbox | clearvoice)
    python setup_env.py --configure                 nur Modell-Ordner / Grafikkarte neu einstellen
    python setup_env.py --slim                      Speicher sparen in bereits vorhandenen Umgebungen
    python setup_env.py --update                    Update abschließen (nach dem Entpacken einer neuen Version;
                                                    wird von update.bat / update.sh aufgerufen)

Optionen: --backend auto|cuda|rocm|cpu   --models-dir PFAD   --non-interactive

Nutzt uv (https://docs.astral.sh/uv/) – wird bei Bedarf nach tools/uv heruntergeladen. Pakete werden
im Cache tools/uv-cache abgelegt, gleiche Dateien (z. B. torch) also nur einmal geladen.
Nur Standardbibliothek, damit das Skript auch vor der Installation der Pakete läuft.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from version import APP_VERSION, UNVERSIONED

BASE_DIR = Path(__file__).resolve().parent
TOOLS_DIR = BASE_DIR / "tools"
CONFIG_FILE = BASE_DIR / "config.json"
IS_WINDOWS = os.name == "nt"
PYTHON_VERSION = "3.12"
INCOMPLETE_MARKER = ".installing"

ENVS = {
    "main": (".venv", "requirements.txt"),
    "parler": (".venv-parler", "requirements-parler.txt"),
    "chatterbox": (".venv-chatterbox", "requirements-chatterbox.txt"),
    "clearvoice": (".venv-clearvoice", "requirements-clearvoice.txt"),
}

# torch-Version je Umgebung und Backend (Index-Suffix auf download.pytorch.org/whl/<suffix>)
TORCH = {
    "main": (["torch==2.8.0", "torchaudio==2.8.0"], {"cuda": "cu128", "cpu": "cpu", "rocm": "rocm6.4"}),
    "parler": (["torch==2.8.0", "torchaudio==2.8.0"], {"cuda": "cu128", "cpu": "cpu", "rocm": "rocm6.4"}),
    "clearvoice": (["torch==2.8.0", "torchaudio==2.8.0", "torchvision==0.23.0"],
                   {"cuda": "cu128", "cpu": "cpu", "rocm": "rocm6.4"}),
    "chatterbox": (["torch==2.8.0", "torchaudio==2.8.0"], {"cuda": "cu128", "cpu": "cpu", "rocm": "rocm6.4"}),
}
# chatterbox-tts verlangt offiziell torch 2.6, läuft aber mit 2.8 (getestet, lädt sogar schneller). Mit derselben
# Version wie die anderen Umgebungen teilt es sich die PyTorch-Dateien (Hardlinks) – spart ca. 5 GB.
TORCH_OVERRIDE = {"chatterbox"}

# AMD-Radeon unter Windows: offizielle ROCm-Pakete von AMD (benötigt Python 3.12 und Treiber 26.2.2+)
AMD_WIN = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"
AMD_WIN_SDK = [AMD_WIN + f for f in (
    "rocm_sdk_core-7.2.1-py3-none-win_amd64.whl", "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl", "rocm-7.2.1.tar.gz")]
AMD_WIN_TORCH = {
    "torch": AMD_WIN + "torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    "torchaudio": AMD_WIN + "torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    "torchvision": AMD_WIN + "torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
}
AMD_WIN_SUPPORTED = re.compile(r"RX\s*(7[6-9]\d\d|9\d\d\d)|AI\s*PRO|PRO\s*W7", re.I)

# Abhängigkeiten, die die Pakete selbst per git angeben – als ZIP-Archiv (kein git nötig)
OVERRIDES = {
    "parler": ["descript-audiotools @ https://github.com/descriptinc/audiotools/archive/"
               "348ebf2034ce24e2a91a553e3171cb00c0c71678.zip"],
    "chatterbox": ["resemble-perth @ https://github.com/resemble-ai/Perth/archive/"
                   "ff1c8ac55a976971245cdd53c18d6131ca00d993.zip"],
}
# Nachträglich ohne Abhängigkeiten installiert (deren Abhängigkeiten werden nicht benötigt)
NO_DEPS = {"main": ["qwen-asr==0.0.6"]}
# Importe, die nach der Installation einmal ausgeführt werden (siehe install_env, Schritt 4).
WARMUP = {
    "main": "import gradio, librosa, transformers, qwen_tts, qwen_asr, snac, onnxruntime, scipy.signal",
    "parler": "import parler_tts, transformers",
    "chatterbox": "import chatterbox.mtl_tts",
    "clearvoice": "import clearvoice",
}

BACKEND_TEXT = {
    "cuda": "NVIDIA-Grafikkarte (CUDA)",
    "rocm": "AMD-Grafikkarte (ROCm)",
    "cpu": "Nur Prozessor (CPU, langsam)",
}


# ---------------------------------------------------------------- Ausgabe
def say(text: str = "") -> None:
    print(text, flush=True)


def step(i: int, n: int, text: str) -> None:
    # Maschinenlesbare Zeile für die Fortschrittsanzeige in der App
    say(f"##STEP {i}/{n} {text}")


def ask(prompt: str, default: str = "") -> str:
    try:
        value = input(prompt).strip()
    except EOFError:
        value = ""
    return value or default


# ---------------------------------------------------------------- Konfiguration
def load_config() -> dict:
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(**values) -> None:
    data = load_config()
    data.update(values)
    CONFIG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_dir(raw: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(raw.strip().strip('"'))))
    return (path if path.is_absolute() else BASE_DIR / path).resolve()


def choose_models_dir(given: str | None, interactive: bool) -> str:
    current = given or load_config().get("models_dir") or "models"
    if given or not interactive:
        return current
    say()
    say("Wo sollen die Modelle gespeichert werden? Je nach Auswahl werden 1–60 GB benötigt.")
    say(f"  Enter = {resolve_dir(current)}")
    while True:
        raw = ask("  Ordner: ", current)
        try:
            path = resolve_dir(raw)
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as e:
            say(f"  Ordner kann nicht verwendet werden ({e}). Bitte einen anderen wählen.")
            continue
        # Innerhalb des Programmordners relativ speichern, damit der Ordner verschiebbar bleibt.
        try:
            return path.relative_to(BASE_DIR).as_posix()
        except ValueError:
            return str(path)


# ---------------------------------------------------------------- Grafikkarte erkennen
def _run(cmd: list[str], timeout: float = 20) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace",
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def detect_backend() -> tuple[str | None, str]:
    """(Backend oder None = unklar, Begründung)."""
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if out and out.strip():
        return "cuda", f"NVIDIA erkannt: {out.strip().splitlines()[0]}"
    if platform.system() == "Darwin":
        return "cpu", "macOS – Apple-GPU wird über die Standard-torch-Pakete genutzt."
    if IS_WINDOWS:
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"])
        if out is None:
            return None, "Grafikkarte konnte nicht ermittelt werden."
        names = [n.strip() for n in out.splitlines() if n.strip()]
    else:
        out = _run(["lspci"])
        if out is None:
            names = []
            if Path("/dev/kfd").exists():
                return "rocm", "AMD-ROCm-Treiber (/dev/kfd) gefunden."
            return None, "Grafikkarte konnte nicht ermittelt werden (lspci fehlt)."
        names = [ln for ln in out.splitlines() if re.search(r"VGA|3D|Display", ln)]
    amd = [n for n in names if re.search(r"AMD|ATI|Radeon", n, re.I)]
    if amd:
        if IS_WINDOWS:
            if any(AMD_WIN_SUPPORTED.search(n) for n in amd):
                return "rocm", f"AMD erkannt: {amd[0]}"
            return None, f"AMD erkannt ({amd[0]}), aber unklar, ob ROCm unter Windows unterstützt wird."
        if Path("/dev/kfd").exists() or shutil.which("rocminfo"):
            return "rocm", f"AMD mit ROCm-Treiber erkannt: {amd[0]}"
        return None, f"AMD erkannt ({amd[0]}), aber kein ROCm-Treiber gefunden."
    if any(re.search(r"NVIDIA", n, re.I) for n in names):
        return None, "NVIDIA-Karte gefunden, aber nvidia-smi fehlt (Treiber installiert?)."
    return "cpu", "Keine unterstützte Grafikkarte gefunden – es wird der Prozessor verwendet."


def choose_backend(given: str, interactive: bool) -> str:
    if given in BACKEND_TEXT:
        return given
    backend, reason = detect_backend()
    say(f"Grafikkarte: {reason}")
    if backend and (not interactive or backend != "cpu"):
        return backend
    if not interactive:
        return load_config().get("torch_backend") or "cpu"
    say()
    say("Welche Hardware soll für die Modelle verwendet werden?")
    say("  1) NVIDIA-Grafikkarte (CUDA) – GeForce GTX 16xx / RTX 20xx oder neuer, aktueller Treiber")
    if IS_WINDOWS:
        say("  2) AMD-Grafikkarte (ROCm)  – Windows: nur Radeon RX 7000/9000, Radeon PRO W7000, AI PRO;")
        say("                                Treiber 26.2.2 oder neuer nötig")
    else:
        say("  2) AMD-Grafikkarte (ROCm)  – ROCm-Treiber muss installiert sein (siehe rocm.docs.amd.com)")
    say("  3) Nur Prozessor (CPU)      – funktioniert immer, aber deutlich langsamer")
    default = {"cuda": "1", "rocm": "2"}.get(backend or "", "3")
    while True:
        choice = ask(f"  Auswahl [{default}]: ", default)
        if choice in ("1", "2", "3"):
            return {"1": "cuda", "2": "rocm", "3": "cpu"}[choice]


# ---------------------------------------------------------------- uv
def uv_exe() -> Path:
    name = "uv.exe" if IS_WINDOWS else "uv"
    local = TOOLS_DIR / "uv" / name
    if local.exists():
        return local
    found = shutil.which("uv")
    if found:
        return Path(found)
    return download_uv()


def download_uv() -> Path:
    machine = platform.machine().lower()
    arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
    if IS_WINDOWS:
        asset = f"uv-{arch}-pc-windows-msvc.zip"
    elif platform.system() == "Darwin":
        asset = f"uv-{arch}-apple-darwin.tar.gz"
    else:
        asset = f"uv-{arch}-unknown-linux-gnu.tar.gz"
    url = f"https://github.com/astral-sh/uv/releases/latest/download/{asset}"
    say(f"Lade uv herunter ({url}) ...")
    data = None
    for attempt in range(1, 7):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                data = resp.read()
            break
        except OSError as e:
            if attempt == 6:
                raise RuntimeError(f"uv konnte nicht heruntergeladen werden: {e}") from e
            say(f"  Verbindungsfehler ({e}) – neuer Versuch in {attempt * 3}s ...")
            time.sleep(attempt * 3)
    target = TOOLS_DIR / "uv"
    target.mkdir(parents=True, exist_ok=True)
    wanted = {"uv.exe", "uvx.exe"} if IS_WINDOWS else {"uv", "uvx"}
    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if Path(member).name in wanted:
                    (target / Path(member).name).write_bytes(zf.read(member))
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for member in tf.getmembers():
                if member.isfile() and Path(member.name).name in wanted:
                    out = target / Path(member.name).name
                    out.write_bytes(tf.extractfile(member).read())
                    out.chmod(0o755)
    exe = target / ("uv.exe" if IS_WINDOWS else "uv")
    if not exe.exists():
        raise RuntimeError("uv konnte nicht entpackt werden.")
    return exe


def uv_env() -> dict:
    env = dict(os.environ)
    env.setdefault("UV_CACHE_DIR", str(TOOLS_DIR / "uv-cache"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(TOOLS_DIR / "python"))
    # Eigenes Python unter tools/python statt eines System-Pythons, das später entfernt werden könnte
    env.setdefault("UV_PYTHON_PREFERENCE", "only-managed")
    env.setdefault("UV_HTTP_TIMEOUT", "300")
    env.setdefault("UV_HTTP_RETRIES", "10")
    # .pyc-Dateien schon bei der Installation erzeugen – sonst dauert der erste Start der App mehrere Minuten.
    env.setdefault("UV_COMPILE_BYTECODE", "1")
    # Fortschrittsbalken nur im Konsolenfenster; in der App wird die Ausgabe zeilenweise ausgewertet.
    if not sys.stdout.isatty():
        env["UV_NO_PROGRESS"] = "1"
    env.pop("VIRTUAL_ENV", None)
    return env


def uv_cache_dir() -> Path:
    return Path(uv_env()["UV_CACHE_DIR"])


def slim_env(env_dir: Path) -> int:
    """Entfernt Dateien, die nur zum Kompilieren von C++-Erweiterungen dienen (unter Windows ca. 2,8 GB je
    PyTorch-Installation, v. a. dnnl.lib). Zur Laufzeit werden nur die .dll-Dateien gebraucht.

    Die Dateien sind per Hardlink mit dem uv-Cache verbunden – Platz wird erst frei, wenn sie auch dort
    gelöscht sind. Gibt die tatsächlich freigegebenen Bytes zurück.
    """
    if not IS_WINDOWS:
        return 0
    dirs = [env_dir / "Lib" / "site-packages" / "torch" / "lib"]
    archive = uv_cache_dir() / "archive-v0"
    # Nur den eigenen Cache anfassen, nicht einen systemweiten uv-Cache anderer Projekte.
    if archive.is_dir() and archive.resolve().is_relative_to(BASE_DIR):
        dirs += list(archive.glob("*/torch/lib"))
    freed = 0
    for d in dirs:
        for f in d.glob("*.lib"):
            try:
                st = f.stat()
                f.unlink()
            except OSError:
                continue
            if st.st_nlink <= 1:
                freed += st.st_size
    return freed


def uv(*args: str, attempts: int = 3) -> None:
    cmd = [str(uv_exe()), *args]
    say("> uv " + " ".join(args))
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(cmd, env=uv_env())
        if proc.returncode == 0:
            return
        if attempt < attempts:  # meist Verbindungsabbrüche – bereits Geladenes liegt im Cache
            say(f"  uv ist fehlgeschlagen (Code {proc.returncode}) – neuer Versuch ({attempt + 1}/{attempts}) ...")
            time.sleep(5 * attempt)
    raise RuntimeError(f"uv ist fehlgeschlagen (Code {proc.returncode}).")


def venv_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


# ---------------------------------------------------------------- Installation
def torch_plan(name: str, backend: str) -> tuple[list[str], list[str], list[str]]:
    """(uv-Argumente für torch, Override-Zeilen, Hinweise)."""
    specs, suffixes = TORCH[name]
    notes: list[str] = []
    overrides: list[str] = specs[:] if name in TORCH_OVERRIDE else []
    if backend == "rocm" and IS_WINDOWS:
        if name == "chatterbox":
            notes.append("Chatterbox gibt es für AMD unter Windows nicht – es wird die CPU-Version installiert.")
            backend = "cpu"
        else:
            pkgs = [AMD_WIN_TORCH[s.split("==")[0]] for s in specs]
            return [*AMD_WIN_SDK, *pkgs], [], notes
    if platform.system() == "Darwin":
        return [*specs], overrides, notes
    return [*specs, "--index-url", f"https://download.pytorch.org/whl/{suffixes[backend]}"], overrides, notes


def install_env(name: str, backend: str) -> None:
    env_name, req_file = ENVS[name]
    env_dir = BASE_DIR / env_name
    py = venv_python(env_dir)
    total = 4
    n = 1

    if name != "main" or not py.exists():
        step(n, total, f"Python-Umgebung {env_name} anlegen")
        if env_dir.exists() and not py.exists():
            shutil.rmtree(env_dir, ignore_errors=True)
        if not py.exists():
            uv("venv", str(env_dir), "--python", PYTHON_VERSION, "--seed")
    n += 1
    # Solange die Markierung existiert, gilt die Umgebung in der App als nicht (vollständig) installiert.
    marker = env_dir / INCOMPLETE_MARKER
    marker.write_text("installation in progress\n", encoding="utf-8")

    torch_args, overrides, notes = torch_plan(name, backend)
    for note in notes:
        say(note)
    step(n, total, "PyTorch installieren (großer Download, einige Minuten)")
    uv("pip", "install", "--python", str(py), *torch_args)
    n += 1

    step(n, total, "Pakete installieren")
    args = ["pip", "install", "--python", str(py), "-r", str(BASE_DIR / req_file)]
    constraints = BASE_DIR / "constraints" / f"{name}.txt"
    if constraints.exists():
        args += ["--constraint", str(constraints)]
    override_lines = overrides + OVERRIDES.get(name, [])
    tmp = None
    if override_lines:
        fd, tmp = tempfile.mkstemp(suffix=".txt", prefix="overrides_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(override_lines) + "\n")
        args += ["--override", tmp]
    try:
        uv(*args)
    finally:
        if tmp:
            os.unlink(tmp)
    for pkg in NO_DEPS.get(name, []):
        uv("pip", "install", "--python", str(py), "--no-deps", pkg)
    n += 1

    step(n, total, "Installation prüfen und ersten Start vorbereiten")
    check = ("import torch; ok = torch.cuda.is_available(); "
             "print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if ok else 'keine (CPU)')")
    out = subprocess.run([str(py), "-c", check], capture_output=True, text=True, errors="replace")
    say((out.stdout or out.stderr).strip())
    if out.returncode != 0:
        raise RuntimeError("torch lässt sich nicht importieren.")
    if WARMUP.get(name):
        # Einmal alles importieren: Windows prüft dabei die neuen Dateien (Virenscanner) – das passiert sonst
        # beim ersten Start der App und dauert dort mehrere Minuten ohne sichtbaren Fortschritt.
        say("Bereite den ersten Start vor (einmalig, 1–3 Minuten) ...")
        subprocess.run([str(py), "-c", WARMUP[name]], capture_output=True, text=True, errors="replace")
    freed = slim_env(env_dir)
    if freed:
        say(f"Nicht benötigte Entwickler-Dateien von PyTorch entfernt: {freed / 1024**3:.1f} GB gespart.")
    marker.unlink(missing_ok=True)
    hashes = load_config().get("env_hashes") or {}
    save_config(env_hashes={**hashes, name: env_hash(name, backend)})


# ---------------------------------------------------------------- Update
def parse_version(text: str) -> tuple:
    return tuple(int(x) for x in (re.findall(r"\d+", text or "") + ["0", "0", "0"])[:3])


def installed_version() -> str:
    return load_config().get("version") or UNVERSIONED


def env_ready(name: str) -> bool:
    env_dir = BASE_DIR / ENVS[name][0]
    return venv_python(env_dir).exists() and not (env_dir / INCOMPLETE_MARKER).exists()


def env_hash(name: str, backend: str) -> str:
    """Fingerabdruck dessen, was in einer Umgebung installiert wird. Ändert er sich durch ein Update
    (andere Paketliste, Versionen oder PyTorch), wird die Umgebung beim Update neu abgeglichen."""
    torch_args, overrides, _ = torch_plan(name, backend)
    h = hashlib.sha256()
    for part in (*torch_args, *overrides, *OVERRIDES.get(name, []), *NO_DEPS.get(name, [])):
        h.update(part.encode() + b"\0")
    for f in (BASE_DIR / ENVS[name][1], BASE_DIR / "constraints" / f"{name}.txt"):
        if f.exists():
            h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def site_packages(env_dir: Path) -> Path | None:
    if IS_WINDOWS:
        return env_dir / "Lib" / "site-packages"
    return next(iter(sorted((env_dir / "lib").glob("python3*/site-packages"))), None)


def own_cache() -> Path | None:
    """Der uv-Cache, falls er zu dieser Installation gehört (systemweite Caches werden nie angefasst)."""
    cache = uv_cache_dir()
    try:
        return cache if cache.is_dir() and cache.resolve().is_relative_to(BASE_DIR) else None
    except OSError:
        return None


def relink_torch(name: str, backend: str) -> None:
    """Installiert PyTorch aus dem gemeinsamen Cache neu, wenn die Dateien eine eigene Kopie sind (z. B. mit pip
    angelegte Umgebungen). Danach teilen sich alle Umgebungen dieselben Dateien per Hardlink."""
    sp = site_packages(BASE_DIR / ENVS[name][0])
    probe = sp / "torch" / "lib" / ("torch_cpu.dll" if IS_WINDOWS else "libtorch_cpu.so") if sp else None
    if own_cache() is None or probe is None or not probe.exists() or probe.stat().st_nlink > 1:
        return
    # Andere torch-Version (z. B. Chatterbox mit 2.6): wird beim Abgleich der Umgebung ohnehin ersetzt.
    if not any(sp.glob(f"torch-{TORCH[name][0][0].split('==')[1]}*.dist-info")):
        return
    torch_args, _, _ = torch_plan(name, backend)
    reinstall = [a for spec in TORCH[name][0] for a in ("--reinstall-package", spec.split("==")[0])]
    say(f"Umgebung '{name}': PyTorch ist eine eigene Kopie – wird mit den anderen Umgebungen zusammengelegt ...")
    uv("pip", "install", "--python", str(venv_python(BASE_DIR / ENVS[name][0])), *reinstall, *torch_args)


def prune_cache() -> int:
    """Löscht Pakete aus dem eigenen uv-Cache, die von keiner Umgebung mehr benutzt werden (z. B. eine alte
    PyTorch-Version nach einem Update). Benutzte Pakete sind per Hardlink verbunden und bleiben erhalten.
    uv lädt Gelöschtes bei Bedarf einfach neu. Gibt die freigegebenen Bytes zurück."""
    cache = own_cache()
    archive = cache / "archive-v0" if cache else None
    if archive is None or not archive.is_dir():
        return 0
    freed = 0
    for entry in archive.iterdir():
        if not entry.is_dir():
            continue
        size, linked = 0, False
        for f in entry.rglob("*"):
            try:
                st = f.lstat()
            except OSError:
                continue
            if f.is_file():
                if st.st_nlink > 1:
                    linked = True
                    break
                size += st.st_size
        if not linked and size >= 1024**2:
            shutil.rmtree(entry, ignore_errors=True)
            freed += size
    return freed


def migrate_1_1_0(backend: str) -> None:
    for name in ENVS:
        if env_ready(name):
            relink_torch(name, backend)


# Versionsabhängige Update-Schritte: (Version, Beschreibung, Funktion oder None). Ein Update führt alle Schritte
# aus, deren Version neuer als die installierte ist – der Reihe nach. Geänderte Paketlisten werden zusätzlich
# automatisch erkannt (env_hash) und brauchen hier keinen eigenen Eintrag.
MIGRATIONS = [
    ("1.1.0", [
        "Chatterbox nutzt dasselbe PyTorch wie die anderen Umgebungen (ca. 5 GB weniger, lädt deutlich schneller)",
        "Parler-TTS lädt ca. 10x schneller (Paket accelerate)",
        "Schnellere Modell-Downloads (Paket hf_xet)",
        "Nicht benötigte PyTorch-Entwicklerdateien werden entfernt (ca. 2,6 GB je Umgebung)",
        "Mit pip angelegte Umgebungen teilen sich PyTorch per Hardlink",
    ], migrate_1_1_0),
]


def run_update(backend: str, interactive: bool) -> None:
    current = installed_version()
    pending = [m for m in MIGRATIONS if parse_version(current) < parse_version(m[0]) <= parse_version(APP_VERSION)]
    hashes = load_config().get("env_hashes") or {}
    envs = [n for n in ENVS if env_ready(n)]
    stale = [n for n in envs if hashes.get(n) != env_hash(n, backend)]
    say(f"Installierte Version: {current}   ->   neue Version: {APP_VERSION}")
    if parse_version(current) > parse_version(APP_VERSION):
        say("Hinweis: Die installierte Version ist neuer als diese Programmdateien (Downgrade).")
    if not pending and not stale:
        save_config(version=APP_VERSION)
        say("Alles ist auf dem neuesten Stand – nichts zu tun.")
        return
    if pending:
        say("Änderungen:")
        for version, notes, _ in pending:
            say(f"  Version {version}:")
            for note in notes:
                say(f"    - {note}")
    say("Python-Umgebungen, die aktualisiert werden: " + (", ".join(ENVS[n][0] for n in stale) or "keine"))
    say()
    if interactive and ask("Update jetzt durchführen? [J/N] ", "N").strip().lower()[:1] not in ("j", "y"):
        raise KeyboardInterrupt
    total = len(pending) + len(stale) + 1
    n = 0
    for version, _, func in pending:
        n += 1
        say()
        say(f"[Update {n}/{total}] Anpassungen für Version {version} ...")
        if func:
            func(backend)
    for name in stale:
        n += 1
        say()
        say(f"[Update {n}/{total}] Umgebung {ENVS[name][0]} aktualisieren ...")
        install_env(name, backend)
    n += 1
    say()
    say(f"[Update {n}/{total}] Aufräumen ...")
    freed = sum(slim_env(BASE_DIR / ENVS[e][0]) for e in envs) + prune_cache()
    say(f"  {freed / 1024**3:.1f} GB Speicher freigegeben.")
    save_config(version=APP_VERSION)
    say()
    say(f"Update auf Version {APP_VERSION} abgeschlossen.")


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Python-Umgebungen der TTS-&-ASR-App einrichten.")
    parser.add_argument("--env", choices=sorted(ENVS), action="append", help="Umgebung(en), die installiert werden")
    parser.add_argument("--backend", default="auto", choices=["auto", *BACKEND_TEXT])
    parser.add_argument("--models-dir")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--configure", action="store_true", help="nur Einstellungen abfragen und speichern")
    parser.add_argument("--slim", action="store_true",
                        help="nicht benötigte PyTorch-Entwicklerdateien aus vorhandenen Umgebungen entfernen")
    parser.add_argument("--update", action="store_true",
                        help="Update abschließen: Umgebungen abgleichen und versionsabhängige Schritte ausführen")
    args = parser.parse_args(argv)
    interactive = not args.non_interactive and sys.stdin.isatty()

    if args.slim:
        freed = sum(slim_env(BASE_DIR / env_name) for env_name, _ in ENVS.values() if (BASE_DIR / env_name).is_dir())
        freed += prune_cache()
        say(f"Fertig – {freed / 1024**3:.1f} GB freigegeben.")
        return 0
    if args.update:
        try:
            backend = load_config().get("torch_backend")
            if backend not in BACKEND_TEXT:
                backend = choose_backend(args.backend, False)
                save_config(torch_backend=backend)
            run_update(backend, interactive)
            say("##DONE")
            return 0
        except KeyboardInterrupt:
            say("Abgebrochen – es wurde nichts verändert." if not interactive else "Abgebrochen.")
            return 130
        except Exception as e:  # noqa: BLE001
            say(f"##ERROR {e}")
            return 1
    envs = args.env or ([] if args.configure else ["main"])
    try:
        cfg = load_config()
        is_new = not CONFIG_FILE.exists()
        if is_new:
            # Neuinstallation: gleich auf dem Stand dieser Version (ältere Installationen haben keine Versionsangabe).
            save_config(version=APP_VERSION)
        if args.configure or "main" in envs or is_new:
            backend = choose_backend(args.backend, interactive)
            models_dir = choose_models_dir(args.models_dir, interactive)
            resolve_dir(models_dir).mkdir(parents=True, exist_ok=True)
            save_config(models_dir=models_dir, torch_backend=backend)
            say(f"Einstellungen gespeichert: Modelle in {resolve_dir(models_dir)}, {BACKEND_TEXT[backend]}")
        else:
            backend = args.backend if args.backend in BACKEND_TEXT else cfg.get("torch_backend", "cpu")
        for name in envs:
            say()
            say(f"=== Umgebung '{name}' ({BACKEND_TEXT[backend]}) ===")
            install_env(name, backend)
        say()
        say("##DONE")
        return 0
    except KeyboardInterrupt:
        say("Abgebrochen.")
        return 130
    except Exception as e:  # noqa: BLE001
        say(f"##ERROR {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
