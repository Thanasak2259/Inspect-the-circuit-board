@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First launch - setting up Python automatically...
  where py >nul 2>&1
  if not errorlevel 1 (
    py -3.12 -m venv .venv 2>nul
    if errorlevel 1 py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
  if errorlevel 1 goto fail
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" -u run.py
if errorlevel 1 goto fail
exit /b 0
:fail
echo Setup or launch failed.
pause
exit /b 1
