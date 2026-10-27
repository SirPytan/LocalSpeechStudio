# LocalSpeechStudio

**A local web interface and one-click installer for running selected open text-to-speech (TTS),
speech-to-text (ASR) and audio cleanup models on your own hardware.**

LocalSpeechStudio is a custom [Gradio](https://www.gradio.app/) interface that bundles several open speech models
behind one consistent UI. It also includes installers that set up Python, PyTorch and all dependencies for you.
Everything runs locally on your machine, and no audio or text leaves your computer.

> [!IMPORTANT]
> **This repository does not contain or distribute any models.** It only provides the interface, the installers
> and the glue code. Models are downloaded on demand from their original sources (mostly Hugging Face) when you
> click **Download & Install** in the app. Each model has its own license and terms of use, which you must read
> and follow yourself (see [Supported models](#supported-models)).

> [!NOTE]
> **Current focus: German.** The model selection, defaults and the user interface (currently German-language
> only) are tuned for German TTS and ASR. Most models are multilingual, so English and many other languages work
> as well; you just choose the language in the respective tab.

## Features

- **Voice Designer**: create a new voice from a text description ("a calm, deep male voice, speaking slowly…").
- **Voice Cloner**: clone a voice from a short reference recording. Designed voices can be saved to a voice library
  and reused with any text.
- **Voice Generator**: read text with built-in speakers, including dedicated German speakers with emotions.
- **Speech-to-Text**: transcribe recordings. The same models can auto-fill the reference transcript for cloning.
- **Audio-Cleanup**: remove noise, separate speakers or enhance recordings, e.g. to clean up cloning references.
- **Model manager**: install, load/unload and uninstall each model with one click. It shows disk size, VRAM usage
  and background tasks with progress bars, and checks for model updates on start.
- **Live GPU panel** with VRAM usage and the currently loaded models.
- **Lightweight installer** (about 100 KB). It downloads [uv](https://docs.astral.sh/uv/), a private Python 3.12,
  PyTorch matching your GPU (CUDA / ROCm / CPU) and all packages, and lets you choose where models are stored.
- **Versioned updates**: extract a new release over the old one and run `update.bat` / `update.sh`. It applies
  all migration steps version by version and keeps your models, voices and settings.
- **Isolated environments**: models with conflicting dependencies (Parler-TTS, Chatterbox, ClearerVoice-Studio)
  run in their own small Python environments as background workers. These share files via hardlinks to save
  disk space.
- **Console commands** in the app window (`help`, `status`, `load`, `unload`, `install`, `uninstall`, `update`,
  `exit`). Closing the window unloads all models.

## Supported models

Licenses are listed as stated on the model pages at the time of writing. Always check the linked page before use,
because some models are **non-commercial** or **gated** (you must accept the license on Hugging Face and store an
access token in the app).

### Text-to-speech

| Category | Model | Languages | Disk | License |
| --- | --- | --- | --- | --- |
| Voice Designer | [Qwen3-TTS 1.7B VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | 10 languages incl. German, English | 4.2 GB | Apache 2.0 |
| Voice Designer | [Parler-TTS Mini Multilingual v1.1](https://huggingface.co/parler-tts/parler-tts-mini-multilingual-v1.1) (+ tokenizer of [flan-t5-large](https://huggingface.co/google/flan-t5-large)) | 8 European languages incl. German, English | 3.5 GB | Apache 2.0 |
| Voice Cloner | [Qwen3-TTS 1.7B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | 10 languages incl. German, English | 4.2 GB | Apache 2.0 |
| Voice Cloner | [Audio8-TTS Preview 0.6B](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.6b) | multilingual incl. German, English | 2.4 GB | Apache 2.0 |
| Voice Cloner | [Chatterbox Multilingual (v3)](https://huggingface.co/ResembleAI/chatterbox) | 23 languages incl. German, English | 3.0 GB | MIT |
| Voice Generator | [Qwen3-TTS 1.7B CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | 10 languages incl. German, English | 4.2 GB | Apache 2.0 |
| Voice Generator | [Kartoffel-Orpheus 3B German Natural](https://huggingface.co/SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1) (+ [SNAC 24 kHz](https://huggingface.co/hubertsiuzdak/snac_24khz) decoder) | German only | 6.2 GB | Llama 3.2 Community (gated) |
| Voice Generator | [Kartoffel-Orpheus 3B German Synthetic](https://huggingface.co/SebastianBodza/Kartoffel_Orpheus-3B_german_synthetic-v0.1) (+ SNAC decoder) | German only | 6.2 GB | Llama 3.2 Community (gated) |

### Speech-to-text

| Model | Languages | Disk | License |
| --- | --- | --- | --- |
| [Audio8-ASR High (ARK-ASR 3B)](https://huggingface.co/Edge0/ARK-ASR-3B) | multilingual | 7.6 GB | Apache 2.0 |
| [Audio8-ASR Medium (ARK-ASR 0.6B)](https://huggingface.co/Edge0/ARK-ASR-0.6B) | multilingual | 2.4 GB | Apache 2.0 |
| [Audio8-ASR Low (0.1B)](https://huggingface.co/Edge0/Audio8-ASR-0.1B) | multilingual | 0.6 GB | CC-BY-NC 4.0 |
| [Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | multilingual incl. German, English | 4.4 GB | Apache 2.0 |
| [Whisper Crisper (CrisperWhisper)](https://huggingface.co/nyrahealth/CrisperWhisper): verbatim, keeps fillers like "uh" | German, English | 3.0 GB | CC-BY-NC 4.0 |
| [Whisper Turbo German](https://huggingface.co/primeline/whisper-large-v3-turbo-german) | German (fine-tuned Whisper) | 1.5 GB | Apache 2.0 |

### Audio cleanup

| Tool | What it does | Disk | License |
| --- | --- | --- | --- |
| [Clear](https://huggingface.co/desert-ant-labs/clear) | Very fast noise removal on the CPU | 47 MB | source-available (see model page) |
| [DeepFilterNet 3](https://github.com/Rikorose/DeepFilterNet) (prebuilt `deep-filter` binary) | Thorough noise suppression on the CPU | 26 MB | MIT / Apache 2.0 |
| [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio): [MossFormer2 SE 48K](https://huggingface.co/alibabasglab/MossFormer2_SE_48K), [FRCRN SE 16K](https://huggingface.co/alibabasglab/FRCRN_SE_16K), [MossFormerGAN SE 16K](https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K), [MossFormer2 SS 16K](https://huggingface.co/alibabasglab/MossFormer2_SS_16K), [MossFormer2 SR 48K](https://huggingface.co/alibabasglab/MossFormer2_SR_48K) | GPU denoising, speaker separation, super-resolution to 48 kHz | 1.4 GB | Apache 2.0 |

Credits go to the respective authors: Alibaba Qwen team, SebastianBodza, Hugging Face (Parler-TTS),
Edge0, Resemble AI, primeline, nyra health, Desert Ant Labs, Alibaba Tongyi Lab and Hendrik Schröter.

## Requirements

- **Windows 10/11 (64-bit)** or **Linux** (x86_64 / ARM64).
- **GPU**: an NVIDIA GPU with a recent driver is strongly recommended (CUDA). AMD GPUs are supported on Linux via
  ROCm. A CPU fallback exists, but most TTS models are then very slow. 8 GB VRAM is enough for most models;
  Kartoffel-Orpheus and Audio8-ASR High need about 6–8 GB.
- **Disk**: at least **16 GB free** for the base installation (Python, PyTorch, packages and download cache), plus
  the models you choose (0.05–7.6 GB each). Models with their own environment (Parler-TTS, Chatterbox,
  ClearerVoice-Studio) need about 0.5–1 GB extra each.
- An internet connection during installation and model downloads. Python does **not** need to be installed.

## Installation

Download the package for your system from the [latest release](../../releases/latest):

| System | Package | Install | Start |
| --- | --- | --- | --- |
| Windows | `TTS-ASR-lokal-windows.zip` | extract, double-click `install.bat` | `start.bat` |
| Linux | `TTS-ASR-lokal-linux.tar.gz` | `tar xzf …`, then `./install.sh` | `./start.sh` |

> [!WARNING]
> **The Linux package is untested so far.** It uses the same installer logic as Windows (uv, CUDA/ROCm/CPU
> detection), but has not been verified on a real Linux machine yet. Feedback and bug reports are welcome.

The installer shows the free disk space and asks for confirmation. It then asks for the **model folder**
(e.g. a drive with plenty of space), detects your GPU and sets everything up. This takes roughly 5–20 minutes
depending on your connection. Re-run it if it gets interrupted; it resumes where it stopped.

After starting, the web UI opens at <http://127.0.0.1:7860>. Keep the console window open while you use it.
On the first start no models are installed yet: open **📦 Modelle verwalten** (model manager) and click
**⬇ Installieren** next to the models you want. Tabs whose model is missing show a **Download & Install** overlay.
The first start after installation can take a few minutes while Windows scans the new files.

## Updating

1. Close the app (close the console window or type `exit`).
2. Extract the new release **into the existing app folder** and overwrite the files.
3. Run **`update.bat`** (Linux: `./update.sh`).

The update script refuses to run while the app is still open. It applies all migrations between your installed
version and the new one, re-syncs changed Python environments, cleans the download cache and stores the new
version. Models, saved voices and settings are kept. The version is shown in the app header. If you forget the
update step, the app shows a warning.

Model updates are separate: the app checks Hugging Face for newer model files on every start and downloads only
changed files. You can disable this in `start.bat` / `start.sh`.

## Uninstalling models

Click the 🗑 button next to a model in the model manager, or type `uninstall <model>` in the console. This
deletes the model files and, if present, the model's own Python environment, and shows how much space was
freed. You can reinstall it at any time.

## Documentation

- [README.de.md](README.de.md): detailed user guide in German (every model, every setting, disk usage,
  console commands, voice library).
- [UPDATE.md](UPDATE.md): manual updates, rebuilding environments, troubleshooting, and how to publish a new
  version (in German).

## Repository layout

| Path | Purpose |
| --- | --- |
| `app.py` | Gradio web UI, model manager, GPU panel, console commands |
| `*_tts.py`, `*_asr.py`, `asr_models.py`, `audio_cleanup.py` | Model wrappers |
| `*_worker.py`, `*_client.py`, `worker_client.py` | Background workers for models in separate environments |
| `setup_env.py` | Installer / updater: uv, Python, PyTorch per GPU backend, environments, migrations |
| `model_updates.py`, `download_models.py` | Model download, update check and manual download CLI |
| `app_config.py`, `version.py` | Settings (`config.json`) and app version |
| `requirements*.txt`, `constraints/` | Pinned package lists per environment |
| `installer/windows`, `installer/linux` | `install`, `start` and `update` scripts that are packaged into the release |
| `installer/build_installers.py` | Builds the release packages into `dist/` |

### Building the release packages

```bash
python installer/build_installers.py          # both packages
python installer/build_installers.py windows  # or: linux
```

This writes `dist/windows/TTS-ASR-lokal-windows.zip` and `dist/linux/TTS-ASR-lokal-linux.tar.gz`. They contain
only the source files and scripts (about 100 KB) and are published as GitHub release assets, not committed.

### Running from a source checkout

Copy the scripts from `installer/windows` (or `installer/linux`) into the repository root, then run
`install.bat` / `install.sh` there. Root-level copies are git-ignored. Alternatively, build a package as shown
above and extract it anywhere.

## Responsible use

Voice cloning can be misused. Only clone voices of people who have given their consent, label synthetic audio
where appropriate, and follow the license and usage terms of each model. Some models (e.g. Chatterbox) embed an
inaudible watermark in their output.

## License

The code in this repository is licensed under the **[GNU General Public License v3.0](LICENSE)**. You may use,
study, share and modify it. If you distribute the software or a modified version, you must do so under the same
license and make the source code available.

This license covers only the code in this repository. The models and third-party packages that the installer
downloads are subject to their own licenses.
