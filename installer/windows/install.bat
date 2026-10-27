@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title TTS ^& ASR (lokal) - Installation

echo ======================================================================
echo  TTS ^& ASR (lokal) - Installation
echo ======================================================================
echo  Installiert uv, Python 3.12 und alle Pakete in diesen Ordner:
echo    %CD%
echo  Modelle werden NICHT jetzt geladen, sondern spaeter per Klick in der App
echo  (Bereich "Modelle verwalten"). Der Speicherort der Modelle wird gleich abgefragt.
echo  Erneut ausfuehren = Installation reparieren / Einstellungen aendern.
echo ======================================================================
echo.

set "NONINTERACTIVE="
echo.%* | findstr /i /c:"--non-interactive" >nul && set "NONINTERACTIVE=1"
set "FREE_GB="
for /f "usebackq delims=" %%G in (`powershell -NoProfile -Command "try { [math]::Floor((New-Object IO.DriveInfo '%CD:~0,3%').AvailableFreeSpace/1GB) } catch { }"`) do set "FREE_GB=%%G"
if not defined FREE_GB set "FREE_GB=?"

echo  ACHTUNG - SPEICHERPLATZ
echo  ----------------------------------------------------------------------
echo  Die Grundinstallation (ohne Modelle) braucht mindestens 16 GB freien
echo  Speicher auf diesem Laufwerk (fuer Downloads und Entpacken). Danach belegt
echo  sie real ca. 6 GB - der Explorer zeigt ca. 11 GB an, weil .venv und
echo  tools\uv-cache dieselben Dateien nutzen und doppelt gezaehlt werden.
echo  Spaeter kommen die Modelle dazu (je 0,05 - 8 GB). Chatterbox, Parler-TTS
echo  und ClearerVoice-Studio brauchen zusaetzlich je ca. 0,5 GB fuer eine
echo  eigene Python-Umgebung. Die Groessen stehen in der App vor jedem Download.
echo.
echo  Freier Speicher auf %CD:~0,2%  %FREE_GB% GB
if "%FREE_GB%"=="?" goto space_checked
if %FREE_GB% GEQ 16 goto space_checked
echo.
echo  !!! Zu wenig freier Speicher - bitte Platz schaffen oder den Ordner
echo  !!! auf ein anderes Laufwerk verschieben und install.bat dort starten.
:space_checked
echo.
echo  Dauer: je nach Internetverbindung ca. 5 - 20 Minuten.
echo  ----------------------------------------------------------------------
if defined NONINTERACTIVE goto confirmed
set "ANSWER="
set /p "ANSWER=Installation starten? [J/N] "
if /i "%ANSWER:~0,1%"=="J" goto confirmed
if /i "%ANSWER:~0,1%"=="Y" goto confirmed
echo Abgebrochen.
pause
goto end
:confirmed
echo.

if not defined UV_CACHE_DIR set "UV_CACHE_DIR=%CD%\tools\uv-cache"
set "UV_PYTHON_INSTALL_DIR=%CD%\tools\python"
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_HTTP_TIMEOUT=300"
set "UV=%CD%\tools\uv\uv.exe"

if exist "%UV%" goto have_uv
echo [1/3] Lade uv (Paketmanager) herunter ...
set "UV_ARCH=x86_64"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "UV_ARCH=aarch64"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $z=Join-Path $env:TEMP 'uv-download.zip'; $u='https://github.com/astral-sh/uv/releases/latest/download/uv-' + $env:UV_ARCH + '-pc-windows-msvc.zip'; for($i=1; $i -le 6; $i++){ try { Invoke-WebRequest -UseBasicParsing -Uri $u -OutFile $z; break } catch { if($i -eq 6){ throw }; Write-Host ('  Verbindungsfehler - neuer Versuch ' + ($i+1) + '/6 ...'); Start-Sleep ($i*3) } }; New-Item -ItemType Directory -Force 'tools\uv' | Out-Null; Expand-Archive -Force $z 'tools\uv'; Remove-Item $z; $d=(Resolve-Path 'tools\uv').Path; Get-ChildItem $d -Recurse -Filter 'uv*.exe' | Where-Object { $_.DirectoryName -ne $d } | Move-Item -Destination $d -Force"
if not exist "%UV%" goto fail
:have_uv
"%UV%" --version

echo.
echo [2/3] Python 3.12 einrichten ...
if exist ".venv\Scripts\python.exe" goto have_venv
"%UV%" venv .venv --python 3.12 --seed
if errorlevel 1 goto fail
:have_venv

echo.
echo [3/3] Einstellungen abfragen und Pakete installieren (PyTorch ca. 3 GB, dauert einige Minuten) ...
".venv\Scripts\python.exe" setup_env.py --env main %*
if errorlevel 1 goto fail
rem Bei einer bestehenden Installation: versionsabhaengige Update-Schritte nachholen.
".venv\Scripts\python.exe" setup_env.py --update --non-interactive
if errorlevel 1 goto fail

echo.
echo ======================================================================
echo  Installation abgeschlossen.
echo  Starten: start.bat doppelklicken. Modelle in der Weboberflaeche unter
echo  "Modelle verwalten" mit dem Button Download ^& Install herunterladen.
echo ======================================================================
if defined NONINTERACTIVE goto end
set "ANSWER="
set /p "ANSWER=Jetzt starten? [J/N] "
if /i "%ANSWER:~0,1%"=="J" start "" "%~dp0start.bat"
if /i "%ANSWER:~0,1%"=="Y" start "" "%~dp0start.bat"
goto end

:fail
echo.
echo Installation fehlgeschlagen - siehe Meldungen oben. Bei Verbindungsfehlern
echo install.bat einfach erneut ausfuehren (bereits Geladenes wird wiederverwendet).
pause
exit /b 1

:end
endlocal
