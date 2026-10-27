# Manuelle Updates

Normalerweise prüft die App beim Start selbst auf neue Modellversionen (siehe README.de.md, „Modell-Updates“).
Diese Anleitung beschreibt, wie man Modelle und Programmpakete von Hand aktualisiert.
Alle Befehle in PowerShell **im App-Ordner** ausführen; die App vorher beenden (`exit` in der Konsole),
weil Windows geladene Modelldateien sperrt. **Linux:** statt `.\.venv\Scripts\python.exe` jeweils
`.venv/bin/python` verwenden.

**Neue App-Version einspielen:** App beenden, ZIP in den App-Ordner entpacken (überschreiben), dann
`update.bat` (Linux: `./update.sh`) ausführen – siehe README.de.md, „App aktualisieren“. Das entspricht
`.\.venv\Scripts\python.exe setup_env.py --update`.

Einzelne fehlende Modelle installiert man am einfachsten in der App (**📦 Modelle verwalten → ⬇ Installieren**)
oder in der Konsole der laufenden App mit `install <modell>`.

## 1. Modelle

```powershell
# Nur prüfen – zeigt pro Modell ✓ aktuell, ⬆ Update verfügbar oder ⬇ fehlt (mit Dateien und Größe)
.\.venv\Scripts\python.exe download_models.py --check

# Alle Modelle aktualisieren bzw. fehlende herunterladen
.\.venv\Scripts\python.exe download_models.py

# Nur eine Gruppe
.\.venv\Scripts\python.exe download_models.py qwen        # 3 Qwen3-TTS-Modelle
.\.venv\Scripts\python.exe download_models.py kartoffel   # Kartoffel Natural + Synthetic + SNAC
.\.venv\Scripts\python.exe download_models.py parler      # Parler-TTS + flan-t5-Tokenizer
.\.venv\Scripts\python.exe download_models.py audio8      # Audio8-TTS Preview 0.6B
.\.venv\Scripts\python.exe download_models.py chatterbox  # Chatterbox Multilingual (nur die benötigten Dateien)
.\.venv\Scripts\python.exe download_models.py asr         # alle Spracherkennungsmodelle (Audio8-ASR, Qwen3-ASR, Whisper)
.\.venv\Scripts\python.exe download_models.py cleanup     # Clear + ClearerVoice-Studio-Modelle
```

- Es werden nur geänderte oder fehlende Dateien geladen; abgebrochene Downloads werden beim nächsten Aufruf
  fortgesetzt. Verbindungsabbrüche (z. B. WinError 10054) werden automatisch mehrfach wiederholt –
  bei Fehlern den Befehl einfach erneut starten.
- `download_models.py` lädt nur Modelldateien. Zusatzumgebungen (Parler, Chatterbox, ClearerVoice-Studio,
  siehe Abschnitt 2) und das DeepFilterNet-Programm richtet die App beim Installieren selbst ein.
- **Gated Modelle (Kartoffel):** auf der Modellseite auf Hugging Face eingeloggt die Lizenz akzeptieren und den
  Token in der App unter **📦 Modelle verwalten → 🔑 Hugging-Face-Token** speichern (alternativ
  `.\.venv\Scripts\hf.exe auth login`).
- Modelle liegen im Modell-Ordner aus `config.json` (`models_dir`, Standard `models`) unter `<Repo-Name>\`.
  Ein Modell komplett neu laden: Ordner löschen und in der App installieren bzw. `download_models.py <gruppe>`.
- Wie wird geprüft? Hugging Face speichert pro Datei unter `<Modell-Ordner>\<Repo>\.cache\huggingface\download\*.metadata`
  den Prüfwert (ETag). Weicht er von dem auf Hugging Face ab, gilt die Datei als veraltet. Fehlt die Metadatei,
  wird die lokale Datei einmalig gehasht.

## 2. Python-Pakete

Die App nutzt getrennte Umgebungen, weil sich die Pakete gegenseitig ausschließen:

| Umgebung | Für | Paketliste |
| --- | --- | --- |
| `.venv` (`main`) | Oberfläche, Qwen3, Kartoffel, Audio8, alle Spracherkennungsmodelle, Clear, DeepFilterNet | `requirements.txt` |
| `.venv-parler` (`parler`) | Parler-TTS (transformers 4.46.1) | `requirements-parler.txt` |
| `.venv-chatterbox` (`chatterbox`) | Chatterbox (transformers 5.x) | `requirements-chatterbox.txt` |
| `.venv-clearvoice` (`clearvoice`) | ClearerVoice-Studio | `requirements-clearvoice.txt` |

Installiert und aktualisiert wird alles mit `setup_env.py` (nutzt [uv](https://docs.astral.sh/uv/) aus
`tools\uv`). Es installiert die passende PyTorch-Version für die Grafikkarte (CUDA/ROCm/CPU aus `config.json`)
und danach die Pakete laut Paketliste; bereits aktuelle Pakete werden übersprungen.

```powershell
# Hauptumgebung aktualisieren (fragt ggf. Modell-Ordner/Grafikkarte; ohne Fragen: --non-interactive)
.\.venv\Scripts\python.exe setup_env.py --env main --non-interactive

# Zusatzumgebung(en) aktualisieren – nur die, die auch installiert sind
.\.venv\Scripts\python.exe setup_env.py --env parler --env chatterbox --env clearvoice

# Grafikkarte (CUDA/ROCm/CPU) oder Modell-Ordner ändern
.\.venv\Scripts\python.exe setup_env.py --configure

# Speicherplatz sparen: nicht benötigte PyTorch-Entwicklerdateien (*.lib, ca. 2,6 GB je Umgebung) entfernen
# (macht setup_env.py bei jeder Installation automatisch; nur für ältere Installationen nötig, App vorher beenden)
.\.venv\Scripts\python.exe setup_env.py --slim
```

Hinweise:
- Feste Versionen stehen in den requirements-Dateien, in `constraints\<umgebung>.txt` und für PyTorch in
  `setup_env.py` (`TORCH`: alle Umgebungen torch 2.8.0). Für eine neuere Version die Nummer dort ändern und
  den Befehl oben ausführen; torch und torchaudio müssen immer dieselbe Version haben. Gleiche Versionen in
  allen Umgebungen sparen Platz, weil uv die Dateien dann nur einmal speichert (Hardlinks).
- Nie parler-tts oder chatterbox-tts in `.venv` installieren – sie ersetzen transformers, dann laufen Qwen3 und
  Audio8 nicht mehr (Audio8 braucht transformers ≥ 4.57, < 5).
- Prüfen, ob eine Umgebung die GPU nutzt (Beispiel Chatterbox):
  `.\.venv-chatterbox\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
- Installierte Versionen anzeigen: `.\tools\uv\uv.exe pip list --python .venv --outdated`

### Neue Version veröffentlichen (für Entwickler)

1. `APP_VERSION` in `version.py` erhöhen (z. B. `1.2.0`).
2. Muss bei bestehenden Installationen einmalig etwas passieren (Dateien verschieben, Einstellungen umbenennen …),
   in `setup_env.py` eine Funktion `migrate_1_2_0()` schreiben und in `MIGRATIONS` eintragen
   (`("1.2.0", ["Änderung 1", …], migrate_1_2_0)` – die Texte zeigt `update.bat` als Änderungsliste an). `--update` führt alle
   Migrationen zwischen installierter und neuer Version der Reihe nach aus. Geänderte requirements-/constraints-
   Dateien oder PyTorch-Versionen brauchen keine Migration – sie werden über die Prüfsumme erkannt.
3. `.\.venv\Scripts\python.exe installer\build_installers.py` baut ZIP und tar.gz neu.

```powershell
# Ablauf von update.bat von Hand
.\.venv\Scripts\python.exe setup_env.py --update                    # mit Rückfragen
.\.venv\Scripts\python.exe setup_env.py --update --non-interactive  # ohne Rückfragen
```

## 3. Umgebung neu aufbauen (wenn etwas kaputt ist)

```powershell
# Zusatzumgebung (Beispiel Chatterbox) – oder Ordner löschen und in der App erneut „Installieren“ klicken
Remove-Item .venv-chatterbox -Recurse -Force
.\.venv\Scripts\python.exe setup_env.py --env chatterbox

# Hauptumgebung: Ordner löschen und den Installer erneut ausführen
Remove-Item .venv -Recurse -Force
.\install.bat          # Linux: rm -rf .venv && ./install.sh
```

Modelle (Modell-Ordner), Stimmen (`voices\`) und Einstellungen (`config.json`) bleiben dabei erhalten.
Ein abgebrochener Aufbau wird erkannt (Datei `.installing` in der Umgebung) und beim nächsten Mal wiederholt.

## 4. Fehlerbehebung

| Problem | Lösung |
| --- | --- |
| Update „verschoben (geladen)“ | Modell war beim Start schon geladen. Entladen und in der Konsole `update` eingeben. |
| `401` / `403` beim Download | Gated Modell: Lizenz akzeptieren und Hugging-Face-Token in der App speichern (siehe oben). |
| Download bricht immer wieder ab | Befehl erneut starten – er setzt fort. Ggf. VPN/Proxy prüfen. |
| Parler startet nicht | `parler_worker.log` ansehen; ggf. `.venv-parler` neu aufbauen (Abschnitt 3). |
| Chatterbox startet nicht / sehr langsam | `chatterbox_worker.log` ansehen; torch-Version prüfen (Abschnitt 2), ggf. `.venv-chatterbox` neu aufbauen. |
| `WinError 127` beim Import von torchaudio | torchaudio passt nicht zu torch: `setup_env.py --env main` erneut ausführen (installiert beide passend). |
| ClearerVoice-Studio startet nicht | `clearvoice_worker.log` ansehen; ggf. `.venv-clearvoice` neu aufbauen (Abschnitt 3). |
| DeepFilterNet fehlt / startet nicht | Ordner `deepfilternet` im Modell-Ordner löschen und in der App erneut installieren. |
| Installation bricht ab | Installer bzw. „Installieren“ erneut starten – Downloads werden fortgesetzt (Cache `tools\uv-cache`). |
| „Xet Storage is enabled … hf_xet package is not installed“ | Langsamer Download-Weg. `setup_env.py --env main` erneut ausführen (installiert `hf_xet`). |
| Parler-TTS lädt über 60 s | In `.venv-parler` fehlt `accelerate`: `setup_env.py --env parler` erneut ausführen. |
| App zeigt „Update nicht abgeschlossen“ (⚠️ in der Überschrift) | Nach dem Entpacken wurde `update.bat` nicht ausgeführt: App beenden, `update.bat` starten. |
| Modell soll weg (Platz sparen) | In der App 🗑 neben dem Modell bzw. in der Konsole `uninstall <modell>`. |
| Ordner wirken sehr groß (Explorer) | `.venv*` und `tools\uv-cache` teilen sich Dateien (Hardlinks) und werden mehrfach gezählt – siehe README.de.md „Speicherplatz“. |
| Automatische Prüfung stört / offline arbeiten | In `start.bat` vor `set QWEN_TTS_AUTO_UPDATE=0` das `rem ` entfernen (Linux: `QWEN_TTS_AUTO_UPDATE=0 ./start.sh`). |
