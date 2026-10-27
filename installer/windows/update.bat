@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title TTS ^& ASR (lokal) - Update

echo ======================================================================
echo  TTS ^& ASR (lokal) - Update abschliessen
echo ======================================================================
echo  WOZU: Nach dem Entpacken einer neuen Version bringt update.bat die
echo  vorhandene Installation auf den Stand der neuen Programmdateien. Die
echo  neuen Dateien allein reichen nicht: geaenderte Python-Pakete und die
echo  Umstellungen zwischen den Versionen werden erst hiermit angewendet.
echo.
echo  SO GEHT EIN UPDATE:
echo    1. App beenden (Konsolenfenster schliessen oder "exit" eingeben).
echo    2. Die neue ZIP-Datei in DIESEN Ordner entpacken, Dateien ersetzen:
echo       %CD%
echo    3. update.bat ausfuehren (dieses Fenster).
echo.
echo  WAS PASSIERT:
echo    - Zeigt installierte und neue Versionsnummer und die Aenderungen an.
echo    - Fuehrt die Update-Schritte jeder neueren Version der Reihe nach aus.
echo    - Gleicht geaenderte Python-Umgebungen ab (laedt nur Neues herunter).
echo    - Raeumt nicht mehr benoetigte Dateien auf (spart Speicherplatz).
echo  Modelle, gespeicherte Stimmen und Einstellungen bleiben erhalten.
echo  Fuer eine NEUE Installation stattdessen install.bat verwenden.
echo ======================================================================
echo.

set "NONINTERACTIVE="
echo.%* | findstr /i /c:"--non-interactive" >nul && set "NONINTERACTIVE=1"

if exist ".venv\Scripts\python.exe" goto have_venv
echo  In diesem Ordner ist noch keine Installation (.venv fehlt).
echo  Bitte zuerst install.bat ausfuehren.
if not defined NONINTERACTIVE pause
exit /b 1
:have_venv

set "APPDIR=%CD%"
:check_running
powershell -NoProfile -Command "$d = $env:APPDIR.TrimEnd('\') + '\'; $n = @((Get-CimInstance Win32_Process).Where({ $_.ExecutablePath -and $_.ExecutablePath.StartsWith($d, 'OrdinalIgnoreCase') })).Count; if ($n) { Write-Host (' Die App laeuft noch (' + $n + ' Prozess(e) aus diesem Ordner).'); exit 1 }"
if not errorlevel 1 goto not_running
echo  Bitte die App beenden (Konsolenfenster schliessen oder "exit" eingeben).
if defined NONINTERACTIVE exit /b 1
pause
goto check_running
:not_running

".venv\Scripts\python.exe" setup_env.py --update %*
if errorlevel 1 goto fail

echo.
echo ======================================================================
echo  Update abgeschlossen. Starten wie gewohnt mit start.bat.
echo ======================================================================
if not defined NONINTERACTIVE pause
goto end

:fail
echo.
echo Update nicht abgeschlossen - siehe Meldungen oben. Bei Verbindungsfehlern
echo update.bat einfach erneut ausfuehren (bereits Geladenes wird wiederverwendet).
if not defined NONINTERACTIVE pause
exit /b 1

:end
endlocal
