@echo off
rem 2644.T morning signal. Run from Windows Task Scheduler at 08:30 on weekdays (see ops/README.md).
chcp 65001 > nul
cd /d "%~dp0.."
".venv\Scripts\python.exe" ops\morning_signal.py %*
if errorlevel 1 (
  echo [2644.T] Failed to build the morning signal. Check the messages above.
  pause
  exit /b 1
)
start "" notepad "ops\signals\latest.md"
