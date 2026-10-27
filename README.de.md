# TTS & ASR (lokal)

*English overview: [README.md](README.md)*

Lokaler Nachbau der Demo [Qwen/Qwen3-TTS](https://huggingface.co/spaces/Qwen/Qwen3-TTS) mit drei Modi,
ergänzt um ein speziell deutsches Modell (Kartoffel-Orpheus), Parler-TTS als zweiten Voice Designer,
Audio8-TTS (Platz 1 der TTS Arena), Chatterbox Multilingual als Alternative zum Klonen und
mehrere Spracherkennungsmodelle sowie Werkzeuge zur Audio-Bereinigung. Die Oberfläche ist in fünf Kategorie-Tabs
gegliedert, jeweils mit Unter-Tabs pro Modell:

| Kategorie → Unter-Tab | Modell |
| --- | --- |
| Voice Designer → Qwen3 Voice Design | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| Voice Designer → Parler-TTS Multilingual | `parler-tts/parler-tts-mini-multilingual-v1.1` + Tokenizer von `google/flan-t5-large` |
| Voice Cloner → Qwen3 Voice Clone | `Qwen3-TTS-12Hz-1.7B-Base` |
| Voice Cloner → Audio8-TTS 0.6B | `Edge0/Audio8-TTS-Preview-0.6b` |
| Voice Cloner → Chatterbox Multilingual | `ResembleAI/chatterbox` (Multilingual, v3) |
| Voice Generator → Qwen3 CustomVoice | `Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| Voice Generator → Kartoffel-Orpheus | `SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1` / `..._synthetic-v0.1` + SNAC-Decoder |
| Speech-to-Text | `Edge0/ARK-ASR-3B` / `ARK-ASR-0.6B` / `Audio8-ASR-0.1B`, `Qwen/Qwen3-ASR-1.7B`, `nyrahealth/CrisperWhisper`, `primeline/whisper-large-v3-turbo-german` |
| Audio-Cleanup → Clear | `desert-ant-labs/clear` |
| Audio-Cleanup → ClearerVoice-Studio | `alibabasglab/MossFormer2_SE_48K`, `FRCRN_SE_16K`, `MossFormerGAN_SE_16K`, `MossFormer2_SS_16K`, `MossFormer2_SR_48K` |
| Audio-Cleanup → DeepFilterNet | vorkompiliertes Programm `deep-filter` 0.5.6 (GitHub Rikorose/DeepFilterNet) |

Alle Modelle außer Kartoffel-Orpheus (nur Deutsch) sind mehrsprachig (inkl. Deutsch). Sie liegen im Modell-Ordner,
der bei der Installation gewählt wird (Standard `models` im App-Ordner, siehe „Installation“). Nach der
Installation ist noch kein Modell vorhanden – jedes wird bei Bedarf direkt in der Oberfläche installiert.
Jeder Tab hat oben einen Button **🤗 Modell auf Hugging Face**; in der Modelltabelle öffnet das kleine Link-Symbol
vor dem Namen die Modellseite.

Alle Tabs sind gleich aufgebaut: Überschrift der Kategorie mit Kurzbeschreibung, links **Eingabe** (bei Klon-Tabs
**Referenzstimme**), rechts **Ausgabe** mit „Generiertes Audio“, einzeiliger Statuszeile und ggf. **💾 Stimme speichern**.
Der Button **🔊 Generieren** ist überall grün; gleiche Felder heißen überall gleich („Text zum Sprechen“, „Sprache“,
„Stimmbeschreibung“, „Referenz-Audio“, „Referenztext“, „Erweiterte Einstellungen“, „Seed (0 = zufällig)“ …).
Gespeicherte Stimmen lassen sich in allen drei Klon-Tabs auswählen, aktualisieren und löschen.

### Deutsch (Kartoffel-Orpheus 3B)

- **Natural**: 19 natürliche Sprecher (Jakob, Anton, Julian, Jan, Alexander, Emil, Ben, Elias, Felix, Jonas, Noah,
  Maximilian, Sophie, Marie, Mia, Maria, Sophia, Lina, Lea), kaum Emotionen.
- **Synthetic**: 4 Sprecher (Martin, Luca, Anne, Emma) mit Emotion (Happy, Sad, Angry, Fear, …);
  Ausrufe wie „haha“, „ughh“, „wow“, „wuhuuu“, „ohhh“ direkt in den Text schreiben.
- Je ca. 6,2 GB VRAM, ca. 2–3 s Rechenzeit pro Sekunde Audio. Lange Texte werden an Satzgrenzen aufgeteilt.
- Lizenz: Llama 3.2 Community License. Die Repos sind „gated“: zum Herunterladen auf Hugging Face die Lizenz
  akzeptieren und unter **📦 Modelle verwalten → 🔑 Hugging-Face-Token** einen Token (Typ „Read“) speichern.
- **Top-p** (Nucleus Sampling): Bei jedem Schritt wählt das Modell nur aus den wahrscheinlichsten Tokens, deren
  Wahrscheinlichkeiten zusammen p ergeben. Niedriger (z. B. 0,8) = stabiler, aber gleichförmiger;
  höher (bis 1,0) = lebendiger, aber eher Versprecher/Artefakte. Standard 0,95.

### Parler-TTS (Voice Design, mehrsprachig)

- Stimme per **englischer** Beschreibung (Geschlecht, Tonhöhe, Tempo, Ausdruck, Raumklang, Aufnahmequalität);
  die Sprache ergibt sich aus dem Text. Für Deutsch trainierte Sprecher mit gleichbleibender Stimme:
  **Nicole, Michelle, Christopher, Megan** (Namen in der Beschreibung nennen, z. B. „Nicole's voice is …“).
- Vorlagen-Dropdown füllt die Beschreibung aus. **Seed**: 0 = zufällig; gleicher Seed + Text + Beschreibung
  ergibt exakt dasselbe Audio. Gefällt eine Stimme → **💾 Stimme speichern** (landet in der Stimmbibliothek
  für Voice Clone).
- Ca. 1,8 GB VRAM, ca. 1 s Rechenzeit pro Sekunde Audio, 44,1 kHz; erstes Laden ca. 50–80 s. Lizenz: Apache 2.0.
- Läuft in einer **eigenen Python-Umgebung** `.venv-parler` (parler-tts braucht transformers 4.46.1, Qwen eine
  neuere Version) als separater Hintergrundprozess. „Entladen“ beendet diesen Prozess. Fehlermeldungen des
  Prozesses stehen in `parler_worker.log`.

### Audio8-TTS (Preview, 0.6B)

- **Ohne Referenz**: zufällige Stimme – der **Seed** legt sie fest (gleicher Seed = gleiche Stimme). Gefällt sie →
  **💾 Stimme speichern**. Lange Texte werden in Abschnitte geteilt; die Stimme des ersten Abschnitts wird für
  alle weiteren beibehalten.
- **Mit Referenz** (linker Bereich „Referenzstimme (optional)“ oder gespeicherte Stimme): Stimmklon. Das **exakte
  Transkript** der Referenz ist Pflicht, sonst klingt das Ergebnis nach einer fremden Stimme. Referenz max. 30 s,
  ideal 5–15 s.
- Ca. 2 GB VRAM (Spitze ca. 3 GB), 44,1 kHz, ca. 1,5 s Rechenzeit pro Sekunde Audio; Laden ca. 20–30 s.
  Lizenz: Apache 2.0. Noch eine Vorabversion – gelegentliche Aussetzer oder falsche Betonungen sind möglich.
- Läuft in der Hauptumgebung `.venv`.

### Chatterbox Multilingual (Clone)

- **Kein Transkript nötig** – nur 5–15 s Referenz-Audio (oder gespeicherte Stimme). Ohne Referenz spricht die
  eingebaute (englische) Standardstimme Deutsch.
- **Ausdrucksstärke** (0,25–2): 0,5 = neutral, höher = emotionaler und schneller.
  **CFG / Tempo** (0–1): niedriger = langsamer und freier. Für ausdrucksstarke Sprache etwa 0,7 / 0,3.
  Ist die Referenz **nicht deutsch**, CFG auf 0 setzen, sonst wird der Akzent übernommen.
- Ca. 3 GB VRAM, 24 kHz, ca. 0,5–1 s Rechenzeit pro Sekunde Audio; Laden ca. 15–35 s (der erste Start nach
  dem Hochfahren dauert länger). Ausgaben enthalten ein unhörbares Wasserzeichen (Perth). Lizenz: MIT.
- Läuft in einer **eigenen Python-Umgebung** `.venv-chatterbox` (braucht transformers 5.x; nutzt dieselbe
  PyTorch-Version 2.8 wie die App und teilt deren Dateien) als Hintergrundprozess; Fehlermeldungen stehen in `chatterbox_worker.log`.

### Speech-to-Text (Spracherkennung)

- Eigener Kategorie-Tab: Audio hochladen oder aufnehmen, Modell im (alphabetisch sortierten) Dropdown
  **Spracherkennungs-Modell** wählen – darunter erscheinen Beschreibung, Größe auf der Festplatte, VRAM und Lizenz,
  der Hugging-Face-Button oben verweist auf das gewählte Modell – dann **📝 Text erkennen**.
  Der Text ist editierbar und kopierbar. Lange Aufnahmen werden automatisch an leisen Stellen in Abschnitte
  von höchstens 28 s geteilt.
- Unter **Voice Cloner → Qwen3 Voice Clone** und **Audio8-TTS 0.6B** füllt **📝 Referenztext aus Audio erkennen** den
  Referenztext aus dem Referenz-Audio aus (Modell und Sprache daneben wählbar); danach prüfen und bei Bedarf
  von Hand korrigieren.
- **Sprache**: „Deutsch“ (Standard) verhindert, dass Whisper/Qwen3-ASR ins Englische übersetzen; „Automatisch
  erkennen“ für gemischte Aufnahmen. Bei den Audio8-Modellen wird das Feld ausgeblendet (sie erkennen die Sprache selbst).
- Modelle (gemessen auf RTX 4080 SUPER, ca. 6–10 s Audio):

  | Modell | Repo | VRAM | Erkennen | Lizenz |
  | --- | --- | --- | --- | --- |
  | Audio8-ASR Hoch (3B) | `Edge0/ARK-ASR-3B` | ca. 7,7 GB | ca. 1 s | Apache 2.0 |
  | Audio8-ASR Medium (0.6B) | `Edge0/ARK-ASR-0.6B` | ca. 2,3 GB | ca. 0,7 s | Apache 2.0 |
  | Audio8-ASR Low (0.1B) | `Edge0/Audio8-ASR-0.1B` | ca. 0,7 GB | ca. 0,4 s | CC-BY-NC 4.0 |
  | Qwen3-ASR 1.7B | `Qwen/Qwen3-ASR-1.7B` | ca. 3,8 GB | ca. 0,9 s | Apache 2.0 |
  | Whisper Crisper (wortgetreu inkl. „äh“) | `nyrahealth/CrisperWhisper` | ca. 2,9 GB | ca. 0,7 s | CC-BY-NC 4.0 |
  | Whisper Turbo German (Standard) | `primeline/whisper-large-v3-turbo-german` | ca. 1,5 GB | ca. 0,7 s | Apache 2.0 |

  CC-BY-NC = nur nicht-kommerzielle Nutzung.
- **Achtung „Audio8-ASR Hoch (3B)“:** übersetzt deutsche Sprache gelegentlich ins Englische und schreibt alles klein
  ohne Satzzeichen.
- Alle laufen in der Hauptumgebung `.venv`. Qwen3-ASR braucht das Paket `qwen-asr`, das mit `--no-deps`
  installiert wird (erledigt `setup_env.py` automatisch), damit es die vorhandene transformers-Version nicht ersetzt.

### Audio-Cleanup (Aufnahmen bereinigen)

- Eigener Kategorie-Tab mit drei Werkzeugen: Audio hochladen oder aufnehmen → **🧹 Bereinigen** → Ergebnis anhören
  und herunterladen. „Lautstärke normalisieren“ hebt leise Aufnahmen an. Ideal, um Referenzaufnahmen für die
  Klon-Tabs vorher zu säubern.
- **Clear** (`desert-ant-labs/clear`): sehr schnell (wenige Zehntelsekunden), läuft auf dem Prozessor, kein VRAM.
  Variante **Studio** filtert maximal, **Natural** klingt natürlicher; „Stärke“ mischt das Original wieder bei.
  Lizenz: source-available (Bedingungen auf der Modellseite beachten).
- **DeepFilterNet**: gründlicher als Clear, ebenfalls auf dem Prozessor. „Maximale Absenkung“ 20–30 dB lässt etwas
  Umgebung hörbar (natürlicher), „Nachfilter“ entfernt Restrauschen stärker. Lizenz: MIT / Apache 2.0.
- **ClearerVoice-Studio** (Alibaba, GPU, eigene Umgebung `.venv-clearvoice`): Rauschunterdrückung mit drei Modellen,
  **Sprecher trennen** (zwei gleichzeitig sprechende Personen → zwei Spuren „Sprecher 1/2“) und **Qualität erhöhen**
  (dumpfe Aufnahmen auf 48 kHz hochrechnen). Erstes Laden ca. 30 s. Lizenz: Apache 2.0.

## Installation

Das Installationspaket enthält nur die Programmdateien (unter 100 KB); Python, alle Pakete und die Modelle werden
bei der Installation bzw. später in der Oberfläche heruntergeladen. Nötig sind eine Internetverbindung und
**mindestens 16 GB freier Speicher** für die Grundinstallation (plus die gewünschten Modelle). Der Installer zeigt
vorher den freien Platz an und startet erst nach Bestätigung mit **J** (Dauer je nach Leitung ca. 5–20 Minuten).

| System | Paket | Installieren | Starten |
| --- | --- | --- | --- |
| Windows 10/11 | `dist\windows\TTS-ASR-lokal-windows.zip` | entpacken → `install.bat` doppelklicken | `start.bat` |
| Linux (x86_64/ARM64) | `dist/linux/TTS-ASR-lokal-linux.tar.gz` | `tar xzf …` → `./install.sh` | `./start.sh` |

Der Installer
1. lädt [uv](https://docs.astral.sh/uv/) nach `tools/uv` und damit ein eigenes Python 3.12 nach `tools/python`
   (ein vorhandenes Python wird nicht benötigt oder verändert),
2. fragt nach dem **Modell-Ordner** (z. B. ein anderes Laufwerk mit viel Platz) und erkennt die Grafikkarte
   (NVIDIA → CUDA, AMD unter Linux → ROCm, sonst CPU),
3. richtet die Hauptumgebung `.venv` inkl. PyTorch ein und speichert die Einstellungen in `config.json`,
4. bereitet den ersten Start vor (Python-Dateien vorkompilieren, alle Bibliotheken einmal laden) und entfernt
   nicht benötigte PyTorch-Entwicklerdateien (Windows: ca. 2,6 GB pro Umgebung).

Bei Abbrüchen (z. B. Verbindungsfehler) den Installer einfach erneut starten – er setzt fort.
Optionen (an `install.bat`/`install.sh` anhängbar): `--models-dir PFAD`, `--backend auto|cuda|rocm|cpu`,
`--non-interactive`. Einstellungen später ändern:
`.venv\Scripts\python.exe setup_env.py --configure` (Linux: `.venv/bin/python setup_env.py --configure`).
Bereits heruntergeladene Modelle beim Ändern des Ordners von Hand mitverschieben.

Nach dem ersten Start öffnet sich **📦 Modelle verwalten** automatisch: dort bei den gewünschten Modellen
**⬇ Installieren** klicken. Tabs, deren Modell fehlt, zeigen statt der Eingabefelder ein Overlay mit
**Download & Install**. Parler-TTS, Chatterbox und ClearerVoice-Studio brauchen zusätzlich eine eigene
Python-Umgebung – auch die wird dabei automatisch eingerichtet (Tabelle und Overlay nennen den Zusatzbedarf).
Vor jedem Download wird geprüft, ob genug Platz frei ist. Fortschritt (Prozent, Dauer, aktuelle Datei bzw.
Installationsschritt, Download-Rate) unter **🖥️ GPU & Hintergrund-Aufgaben** – die App bleibt währenddessen bedienbar.

### Speicherplatz

| Teil | real belegt | Explorer zeigt |
| --- | --- | --- |
| Grundinstallation (`.venv`, `tools\uv-cache`, Python) | ca. 6 GB | ca. 11 GB |
| + Parler-TTS-, Chatterbox- bzw. ClearerVoice-Umgebung | je ca. 0,5–1 GB | je ca. 5,5 GB |
| Modelle (Modell-Ordner) | je 0,05–8 GB | gleich |

Die Umgebungen und der Download-Zwischenspeicher `tools\uv-cache` teilen sich dieselben Dateien (Hardlinks);
der Explorer zählt sie mehrfach. Den Cache nicht löschen – das spart kaum Platz und bremst spätere Installationen.
Ältere Installationen werden beim Aktualisieren mit `update.bat` automatisch verkleinert (siehe
„App aktualisieren“). Unter Linux enthält PyTorch keine solchen
Dateien; dort belegt die Grundinstallation je nach Grafikkarte ca. 6–10 GB.

**Pakete neu bauen** (nach Änderungen am Code): `.venv\Scripts\python.exe installer\build_installers.py`
→ `dist\windows\…zip` und `dist\linux\…tar.gz`; die Vorlagen der Start-/Installationsskripte liegen in
`installer\windows` bzw. `installer/linux`.

## Starten

`start.bat` doppelklicken (Linux: `./start.sh`) – die Oberfläche öffnet sich unter http://127.0.0.1:7860.
Der erste Start nach der Installation kann länger dauern (Windows prüft die neuen Programmdateien); die Konsole
zeigt jeden Startschritt und meldet alle 15 s, dass sie noch arbeitet.
Das Konsolenfenster muss offen bleiben, solange die Seite genutzt wird; wird es geschlossen,
zeigt der Browser „Connection to the server was lost“. Läuft die App bereits, öffnet `start.bat` nur den Browser.

## App aktualisieren (neue Version)

Die Version steht in der Überschrift der Weboberfläche und beim Start in der Konsole (`version.py`).

1. App beenden (Konsolenfenster schließen oder `exit`).
2. Das neue ZIP (Linux: tar.gz) **in den bisherigen App-Ordner entpacken** und vorhandene Dateien überschreiben.
3. **`update.bat`** doppelklicken (Linux: `./update.sh`).

`update.bat` erklärt beim Start, was es tut, bricht ab, solange die App noch läuft, und führt dann
`setup_env.py --update` aus: Es arbeitet die Änderungen **Version für Version** ab (Migrationen, z. B. 1.0.0 → 1.1.0:
doppelte PyTorch-Kopien durch geteilte Dateien ersetzen), bringt alle installierten Python-Umgebungen auf den
Stand der neuen Paketlisten (nur geänderte Umgebungen; erkannt über Prüfsummen in `config.json`), räumt den
Download-Zwischenspeicher auf und speichert die neue Version in `config.json`. Modelle, Stimmen und Einstellungen
bleiben erhalten. Wurde `update.bat` vergessen, zeigt die App beim Start einen Hinweis (⚠️ in der Überschrift).

## Modell-Updates

Beim Start prüft die App im Hintergrund alle vorhandenen Modelle auf Hugging Face. Gibt es eine neuere Version,
werden nur die geänderten Dateien heruntergeladen (Fortschritt in der Konsole und in der Modelltabelle);
das Modell ist währenddessen gesperrt. Geladene Modelle werden nicht angetastet, sondern beim nächsten
`update` aktualisiert. Ohne Internet startet die App ganz normal.

- Abschalten: in `start.bat` bei `rem set QWEN_TTS_AUTO_UPDATE=0` das `rem ` entfernen
  (Linux: `QWEN_TTS_AUTO_UPDATE=0 ./start.sh`).
- Manuell in der Konsole: `update`. Ohne laufende App: siehe [UPDATE.md](UPDATE.md).

## Modelle verwalten

- **Weboberfläche**:
  - **🖥️ GPU & Hintergrund-Aufgaben** (aufgeklappt): Grafikspeicher-Belegung live (alle 0,5 s), Anteil dieser App und die
    **geladenen Modelle mit ihrer tatsächlichen VRAM-Größe** (⏳ = wird gerade geladen). Laufen Installationen,
    Updates oder Ladevorgänge, erscheinen sie darunter mit Fortschrittsbalken, Dauer, Details und Download-Rate
    (gestreift = Schritt ohne genauen Fortschritt, z. B. Einrichten einer Python-Umgebung). Darunter die Statuszeile
    (beim Laden mit Ladebalken, geschätzt anhand der letzten Ladezeit) und **⏏ Alle Modelle entladen**.
  - **📦 Modelle verwalten** (zugeklappt, beim allerersten Start ohne Modelle aufgeklappt): Modelltabelle nach
    Kategorien (Voice Designer, Voice Cloner, Voice Generator, Speech-to-Text, Audio-Cleanup), innerhalb jeder
    Kategorie alphabetisch (Audio8-ASR in der Reihenfolge Hoch, Medium, Low). Link-Symbol (öffnet Hugging Face),
    Name mit Kurzbeschreibung, Größe auf der Festplatte, VRAM (fett = gemessen, „≈“ = Schätzung bzw. letzter
    Messwert), Status und Button: **⬇ Installieren** (fehlt; Download-Größe steht dabei, Fortschritt in der Statusspalte),
    ▶ Laden / ⏏ Entladen, bei Clear und DeepFilterNet „sofort bereit“ (laufen ohne Laden auf dem Prozessor).
    Daneben **🗑 Deinstallieren** (mit Rückfrage): löscht die Modelldateien und – falls vorhanden – die eigene
    Python-Umgebung des Modells (Parler-TTS, Chatterbox, ClearerVoice-Studio) und zeigt den freigegebenen Platz an;
    über ⬇ Installieren lässt es sich jederzeit wieder einrichten. Gemeinsam genutzte Dateien (z. B. Kartoffels
    SNAC-Decoder, Parlers Tokenizer) bleiben erhalten, solange ein anderes Modell sie braucht.
    Darunter **🔑 Hugging-Face-Token** für zugangsbeschränkte Modelle (Kartoffel). In schmalen Fenstern werden Festplatte- und
    Status-Spalte ausgeblendet.
- **Konsole** (`help` zeigt alle Befehle):

  | Befehl | Wirkung |
  | --- | --- |
  | `status` | Modelle (Größe, Status) und GPU-Speicher anzeigen |
  | `load design\|clone\|tts\|natural\|synthetic\|parler\|audio8\|chatterbox\|whisper\|qwen-asr\|crisper\|asr-high\|asr-medium\|asr-low\|clearvoice` | Modell laden |
  | `unload <Modell wie oben>\|all` | Modell(e) entladen |
  | `install <Modell wie oben>` (auch `clear`, `deepfilter`) | Fehlendes Modell herunterladen und installieren |
  | `uninstall <Modell wie oben>` (auch `clear`, `deepfilter`) | Modelldateien löschen, um Platz freizugeben (mit Rückfrage) |
  | `update` | Modelle auf Hugging Face auf neue Versionen prüfen und herunterladen |
  | `open` | Weboberfläche im Browser öffnen |
  | `exit` | Alle Modelle entladen und beenden |

  Beim Schließen des Fensters oder mit Strg+C werden die Modelle ebenfalls entladen.

## Stimmen speichern

- **Qwen3 Voice Design / Parler-TTS / Audio8-TTS:** Nach dem Generieren unten einen Namen eingeben → **💾 Stimme speichern**.
- **Qwen3 Voice Clone:** Eine hochgeladene Referenz (Audio + Text) lässt sich ebenso mit **💾 Als Stimme speichern** sichern.
- Gespeicherte Stimmen unter **Voice Cloner** (Qwen3 Voice Clone, Audio8-TTS, Chatterbox) bei
  **Gespeicherte Stimme** auswählen – Referenz-Audio, -Text und Sprache werden automatisch ausgefüllt; dann
  beliebigen Zieltext eingeben und generieren.
- Ablage: Ordner `voices\<Name>\` im App-Ordner mit `referenz.wav` und `stimme.json` (Referenztext, Sprache, Beschreibung).
  Zum Sichern/Teilen einfach den Ordner kopieren.

## Hinweise

- Modelle werden beim ersten Gebrauch geladen; standardmäßig bleiben max. 2 gleichzeitig im VRAM
  (Umgebungsvariable `QWEN_TTS_MAX_LOADED`). Zusätzlich wird vor dem Laden geprüft, ob danach noch mind. 3 GB VRAM
  frei bleiben (`QWEN_TTS_VRAM_HEADROOM_GB`) – sonst wird das am längsten ungenutzte Modell entladen. Läuft der
  VRAM voll, lagert Windows in den Arbeitsspeicher aus und die Generierung wird extrem langsam.
- Referenz-Audio für die Klon-Tabs und Audio-Cleanup: WAV, MP3, FLAC, OGG (Vorbis/Opus), AIFF, CAF (ffmpeg ist nicht nötig).
  M4A/AAC, WMA und Videodateien werden nicht unterstützt – vorher in WAV/MP3 umwandeln.
- Hinweise zu fehlendem flash-attn bzw. SoX werden unterdrückt: beides nutzt nur der 25Hz-Tokenizer,
  nicht die hier verwendeten 12Hz-Modelle (diese laufen über PyTorch SDPA).
- Die Stilanweisung unter „Voice Generator → Qwen3 CustomVoice“ wirkt nur subtil (Emotion, Tempo, Tonhöhe), da das Modell die
  Sprecherstimme beibehält. Deutliches Flüstern oder Schreien gelingt mit „Voice Designer“ (Beschreibung z. B.
  „A young man whispering very softly“). Englische Anweisungen wirken am zuverlässigsten.
