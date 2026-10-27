@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto run
echo Die App ist noch nicht installiert - bitte zuerst install.bat ausfuehren.
pause
exit /b 1

:run
set PYTHONIOENCODING=utf-8
rem Automatische Modell-Updates beim Start abschalten: naechste Zeile mit "set QWEN_TTS_AUTO_UPDATE=0" aktivieren
rem set QWEN_TTS_AUTO_UPDATE=0
title TTS ^& ASR (lokal)
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
