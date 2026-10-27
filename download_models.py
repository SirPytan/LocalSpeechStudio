"""Lädt fehlende Modelle herunter bzw. aktualisiert sie auf die neueste Version.

    python download_models.py               alle Modelle laden/aktualisieren
    python download_models.py kartoffel     nur eine Gruppe (qwen | kartoffel | parler | audio8 | chatterbox | asr | cleanup)
    python download_models.py --check       nur prüfen, nichts herunterladen

Es werden nur fehlende oder auf Hugging Face geänderte Dateien geladen; abgebrochene Downloads
werden fortgesetzt. Ziel ist der in config.json eingestellte Modell-Ordner. Die Kartoffel-Modelle sind
"gated": Lizenz auf huggingface.co akzeptieren und den Token in der App unter „Modelle verwalten →
Hugging-Face-Token“ speichern (oder einmalig `hf auth login` aus der .venv ausführen).
Python-Umgebungen und das DeepFilterNet-Programm installiert dieses Skript nicht – dafür die App
(„Download & Install“ bzw. Konsolenbefehl `install <model>`) oder setup_env.py nutzen.
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import model_updates as mu  # noqa: E402


def main(argv) -> int:
    check_only = "--check" in argv
    groups = [a for a in argv if not a.startswith("-")] or mu.GROUPS
    unknown = set(groups) - set(mu.GROUPS)
    if unknown:
        print(f"Unbekannte Gruppe(n): {', '.join(sorted(unknown))}. Erlaubt: {', '.join(mu.GROUPS)}")
        return 2

    failed = 0
    for repo in (r for r in mu.REPOS if r.group in groups):
        try:
            update = mu.check(repo)
        except Exception as e:  # noqa: BLE001
            failed += 1
            if "GatedRepo" in type(e).__name__:
                print(f"✗ {repo.repo_id}: kein Zugriff – Lizenz auf {repo.url} akzeptieren und "
                      "Hugging-Face-Token in der App speichern (oder 'hf auth login').")
            else:
                print(f"✗ {repo.repo_id}: Prüfung fehlgeschlagen ({type(e).__name__}: {e})")
            continue
        if not update.needed:
            print(f"✓ {repo.repo_id}: aktuell")
            continue
        new = not repo.local_dir.exists()
        print(f"{'⬇ Neu' if new else '⬆ Update'}: {repo.repo_id} – {len(update.files)} Datei(en), "
              f"{mu.format_bytes(update.total_bytes)} (Stand {update.last_modified:%Y-%m-%d})")
        for name, size in update.files:
            print(f"    {name} ({mu.format_bytes(size)})")
        if check_only:
            continue

        last = [0.0]

        def progress(done, total, name):
            if time.time() - last[0] >= 2 or done >= total:
                last[0] = time.time()
                print(f"    {done / total * 100:5.1f} %  {mu.format_bytes(done)} / {mu.format_bytes(total)}  ({name})",
                      flush=True)

        try:
            mu.download(update, progress)
            print(f"✓ {repo.repo_id}: fertig")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"✗ {repo.repo_id}: Download fehlgeschlagen ({type(e).__name__}: {e}) – einfach erneut starten.")

    print("Fertig." if not failed else f"Fertig mit {failed} Fehler(n).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
