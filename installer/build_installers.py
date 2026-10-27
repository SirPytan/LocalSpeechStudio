"""Baut die Installationspakete (ohne Modelle, Python-Umgebungen und Pakete – das lädt der Installer).

    python installer/build_installers.py            -> dist/windows/*.zip und dist/linux/*.tar.gz
    python installer/build_installers.py windows    nur ein Paket (windows | linux)

Inhalt: App-Quellcode, requirements*.txt, constraints/, README*.md, UPDATE.md, LICENSE und die Skripte aus
installer/windows bzw. installer/linux. Nicht enthalten: .venv*, models/, voices/, tools/, config.json, Logs.
"""
from __future__ import annotations

import io
import sys
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DIST = ROOT / "dist"
APP_NAME = "TTS-ASR-lokal"

PATTERNS = ["*.py", "requirements*.txt", "constraints/*.txt", "README*.md", "UPDATE.md", "LICENSE"]
SKIP = {"build_installers.py"}


def app_files() -> list[Path]:
    files = sorted({p for pat in PATTERNS for p in ROOT.glob(pat) if p.is_file() and p.name not in SKIP})
    if not (ROOT / "app.py").exists() or (ROOT / "app.py").stat().st_size == 0:
        raise SystemExit("app.py fehlt oder ist leer – Abbruch.")
    return files


def text(path: Path, eol: str) -> bytes:
    data = path.read_bytes().decode("utf-8").lstrip("\ufeff").replace("\r\n", "\n")
    return data.replace("\n", eol).encode("utf-8")


def entries(platform: str) -> list[tuple[str, bytes, bool]]:
    """(Pfad im Archiv, Inhalt, ausführbar)."""
    out = [(f"{APP_NAME}/{p.relative_to(ROOT).as_posix()}", text(p, "\n"), False) for p in app_files()]
    for script in sorted((HERE / platform).iterdir()):
        eol = "\r\n" if script.suffix.lower() in (".bat", ".cmd") else "\n"
        out.append((f"{APP_NAME}/{script.name}", text(script, eol), script.suffix == ".sh"))
    return out


def build_windows() -> Path:
    target = DIST / "windows" / f"{APP_NAME}-windows.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.localtime()[:6]
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, data, _ in entries("windows"):
            zf.writestr(zipfile.ZipInfo(name, stamp), data, zipfile.ZIP_DEFLATED)
    return target


def build_linux() -> Path:
    target = DIST / "linux" / f"{APP_NAME}-linux.tar.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with tarfile.open(target, "w:gz", compresslevel=9) as tf:
        folder = tarfile.TarInfo(APP_NAME)
        folder.type, folder.mode, folder.mtime = tarfile.DIRTYPE, 0o755, now
        tf.addfile(folder)
        for name, data, executable in entries("linux"):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o755 if executable else 0o644, now
            tf.addfile(info, io.BytesIO(data))
    return target


def main(argv: list[str]) -> int:
    builders = {"windows": build_windows, "linux": build_linux}
    wanted = argv or list(builders)
    for name in wanted:
        if name not in builders:
            print(f"Unbekannt: {name} (windows | linux)")
            return 2
        path = builders[name]()
        print(f"{name:<8} {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
