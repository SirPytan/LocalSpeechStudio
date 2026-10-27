"""Modell-Register sowie Update-Prüfung und -Download über Hugging Face.

Für jede Datei eines Modells wird der lokale ETag (aus models/<name>/.cache/huggingface/download/*.metadata)
mit dem aktuellen Stand auf Hugging Face verglichen. Nur geänderte oder neue Dateien werden geladen.
Genutzt von app.py (automatisches Update beim Start) und download_models.py (manuell).
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import filter_repo_objects

from app_config import MODELS_DIR

DEFAULT_IGNORE = ("*.mp4", "*.jpeg", "*.jpg", "*.png", "*.gif")


@dataclass(frozen=True)
class Repo:
    repo_id: str
    group: str
    used_by: tuple  # Modell-Schlüssel in app.py, die dieses Repo benötigen
    allow: tuple | None = None
    ignore: tuple = DEFAULT_IGNORE
    subdir: str = ""  # Unterordner im Modell-Ordner (z. B. für ClearVoice-Checkpoints)
    approx_gb: float = 0.0  # ungefähre Downloadgröße (Anzeige, solange das Modell fehlt)

    @property
    def local_dir(self) -> Path:
        return MODELS_DIR / self.subdir / self.repo_id.split("/")[1]

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"


_CV_FILES = ("last_best_checkpoint", "last_best_checkpoint.pt", "last_best_checkpoint_g.pt",
             "last_best_checkpoint_m.pt", "README.md")

REPOS = [
    Repo("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "qwen", ("VoiceDesign",), approx_gb=4.5),
    Repo("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "qwen", ("Base",), approx_gb=4.5),
    Repo("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "qwen", ("CustomVoice",), approx_gb=4.5),
    Repo("SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1", "kartoffel", ("KartoffelNatural",), approx_gb=6.6),
    Repo("SebastianBodza/Kartoffel_Orpheus-3B_german_synthetic-v0.1", "kartoffel", ("KartoffelSynthetic",), approx_gb=6.6),
    Repo("hubertsiuzdak/snac_24khz", "kartoffel", ("KartoffelNatural", "KartoffelSynthetic"), approx_gb=0.08),
    Repo("parler-tts/parler-tts-mini-multilingual-v1.1", "parler", ("Parler",), approx_gb=3.8),
    # Nur der Tokenizer für die Stimmbeschreibung (keine Gewichte).
    Repo("google/flan-t5-large", "parler", ("Parler",), allow=("*.json", "spiece.model")),
    Repo("Edge0/Audio8-TTS-Preview-0.6b", "audio8", ("Audio8",), approx_gb=2.6),
    # Nur die Dateien für Chatterbox Multilingual V3 (das Repo enthält auch ältere/englische Varianten).
    Repo("ResembleAI/chatterbox", "chatterbox", ("Chatterbox",),
         allow=("ve.pt", "t3_mtl23ls_v3.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json",
                "conds.pt", "Cangjie5_TC.json", "tokenizer.json", "mtl_tokenizer.json"), approx_gb=3.2),
    # Spracherkennung (Transkript für Klon-Referenzen und eigener Tab)
    Repo("primeline/whisper-large-v3-turbo-german", "asr", ("ASR-WhisperDE",),
         ignore=DEFAULT_IGNORE + ("training_args.bin",), approx_gb=1.6),
    Repo("Qwen/Qwen3-ASR-1.7B", "asr", ("ASR-Qwen3",), approx_gb=4.7),
    Repo("nyrahealth/CrisperWhisper", "asr", ("ASR-Crisper",),
         ignore=DEFAULT_IGNORE + ("training_args.bin", "trainer_state.json", "*_results.json", ".eval_results/*"),
         approx_gb=3.2),
    Repo("Edge0/ARK-ASR-3B", "asr", ("ASR-3B",), approx_gb=8.1),
    Repo("Edge0/ARK-ASR-0.6B", "asr", ("ASR-0.6B",), approx_gb=2.6),
    Repo("Edge0/Audio8-ASR-0.1B", "asr", ("ASR-0.1B",), approx_gb=0.7),
    # Audio-Bereinigung
    Repo("desert-ant-labs/clear", "cleanup", ("Clear",),
         allow=("clear-studio.onnx", "clear-natural.onnx", "README.md", "THIRD_PARTY_NOTICES.md"), approx_gb=0.05),
    Repo("alibabasglab/MossFormer2_SE_48K", "cleanup", ("ClearVoice",), allow=_CV_FILES,
         subdir="clearvoice/checkpoints", approx_gb=0.22),
    Repo("alibabasglab/FRCRN_SE_16K", "cleanup", ("ClearVoice",), allow=_CV_FILES,
         subdir="clearvoice/checkpoints", approx_gb=0.16),
    Repo("alibabasglab/MossFormerGAN_SE_16K", "cleanup", ("ClearVoice",), allow=_CV_FILES,
         subdir="clearvoice/checkpoints", approx_gb=0.04),
    Repo("alibabasglab/MossFormer2_SS_16K", "cleanup", ("ClearVoice",), allow=_CV_FILES,
         subdir="clearvoice/checkpoints", approx_gb=0.67),
    Repo("alibabasglab/MossFormer2_SR_48K", "cleanup", ("ClearVoice",), allow=_CV_FILES,
         subdir="clearvoice/checkpoints", approx_gb=0.44),
]
GROUPS = sorted({r.group for r in REPOS})


@dataclass
class UpdateInfo:
    repo: Repo
    revision: str
    files: list = field(default_factory=list)  # [(dateiname, bytes)]
    last_modified: object = None

    @property
    def total_bytes(self) -> int:
        return sum(size for _, size in self.files)

    @property
    def needed(self) -> bool:
        return bool(self.files)


def _meta_path(repo: Repo, filename: str) -> Path:
    return repo.local_dir / ".cache" / "huggingface" / "download" / f"{filename}.metadata"


def _remote_etag(sibling) -> str | None:
    lfs = sibling.lfs
    if lfs:
        return getattr(lfs, "sha256", None) or (lfs.get("sha256") if isinstance(lfs, dict) else None)
    return sibling.blob_id


def _local_etag(repo: Repo, filename: str) -> str | None:
    try:
        lines = _meta_path(repo, filename).read_text(encoding="utf-8").splitlines()
        return lines[1].strip() if len(lines) >= 2 else None
    except OSError:
        return None


def _file_hash(path: Path, lfs: bool) -> str:
    """sha256 (LFS-Dateien) bzw. Git-Blob-SHA1 (normale Dateien) – entspricht dem ETag auf Hugging Face."""
    h = hashlib.sha256() if lfs else hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_meta(repo: Repo, filename: str, commit: str, etag: str) -> None:
    path = _meta_path(repo, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{commit}\n{etag}\n{time.time()}\n", encoding="utf-8")


def _model_info(repo: Repo, timeout: float, attempts: int):
    for attempt in range(1, attempts + 1):
        try:
            return HfApi().model_info(repo.repo_id, files_metadata=True, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            if attempt == attempts or "GatedRepo" in type(e).__name__ or "NotFound" in type(e).__name__:
                raise
            time.sleep(1.5 * attempt)


def check(repo: Repo, timeout: float = 15, attempts: int = 4) -> UpdateInfo:
    """Ermittelt, welche Dateien fehlen oder sich auf Hugging Face geändert haben."""
    info = _model_info(repo, timeout, attempts)
    wanted = set(filter_repo_objects(
        [s.rfilename for s in info.siblings], allow_patterns=repo.allow, ignore_patterns=repo.ignore,
    ))
    result = UpdateInfo(repo, info.sha, last_modified=info.last_modified)
    for s in info.siblings:
        if s.rfilename not in wanted:
            continue
        remote = _remote_etag(s)
        local_file = repo.local_dir / s.rfilename
        if not local_file.exists():
            result.files.append((s.rfilename, s.size or 0))
            continue
        local = _local_etag(repo, s.rfilename)
        if local is None and remote:
            # Ohne Metadaten (z. B. manuell kopiert): Datei einmalig hashen und Ergebnis merken.
            local = _file_hash(local_file, bool(s.lfs))
            if local == remote:
                _write_meta(repo, s.rfilename, info.sha, remote)
        if local != remote:
            result.files.append((s.rfilename, s.size or 0))
    return result


def download(update: UpdateInfo, on_progress=None, attempts: int = 15) -> None:
    """Lädt die geänderten Dateien (mit Wiederholungen bei Verbindungsabbrüchen).

    on_progress(geladene_bytes, gesamt_bytes, dateiname) wird etwa jede Sekunde aufgerufen.
    """
    repo = update.repo
    total = max(update.total_bytes, 1)
    done = 0
    cache_dir = repo.local_dir / ".cache" / "huggingface" / "download"
    for filename, size in update.files:
        stop = threading.Event()

        def poll(name=filename, base=done):
            while not stop.wait(1.0):
                partial = sum(p.stat().st_size for p in cache_dir.rglob("*.incomplete")) if cache_dir.exists() else 0
                if on_progress:
                    on_progress(min(base + partial, total), total, name)

        poller = threading.Thread(target=poll, daemon=True)
        poller.start()
        # Veraltete Metadaten entfernen, sonst überspringt hf_hub_download die Datei bei gleichem Commit.
        _meta_path(repo, filename).unlink(missing_ok=True)
        try:
            for attempt in range(1, attempts + 1):
                try:
                    hf_hub_download(repo.repo_id, filename, local_dir=repo.local_dir, revision=update.revision)
                    break
                except Exception as e:  # noqa: BLE001
                    if "GatedRepo" in type(e).__name__ or "RepositoryNotFound" in type(e).__name__:
                        raise
                    if attempt == attempts:
                        raise
                    time.sleep(min(3 * attempt, 30))
        finally:
            stop.set()
            poller.join()
        done += size
        if on_progress:
            on_progress(done, total, filename)


def format_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.0f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"
