@echo off
REM One-click demo launcher: stop stale servers, health check, then start the server.
REM Run scripts\open_firewall.ps1 ONCE first (right-click > Run with PowerShell) so phones can connect.
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
set PYTHONIOENCODING=utf-8

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop_server.ps1"

echo === Pre-flight check ===
python scripts\health_check.py
if errorlevel 1 (
  echo.
  echo Fix the FAIL items above ^(Ollama must be running: "ollama serve"^), then run this again.
  pause
  exit /b 1
)

echo.
echo === Starting server ^(wait ~30 s for model warm-up; Ctrl+C to stop^) ===
echo Open the https://... address printed below on this laptop or on your phone.
python -m disaster_assistant.api.server
echo.
echo The server stopped. Scroll up for the reason, then press a key to close.
pause
