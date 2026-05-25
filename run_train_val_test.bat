@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
)

if not defined PYTHON_EXE if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

if not defined PYTHON_EXE (
    py -3 --version >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=py -3"
)

if not defined PYTHON_EXE (
    python --version >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo Python executable was not found.
    echo Install Python, create .venv, or edit this file to point to python.exe.
    pause
    exit /b 1
)

echo Using Python: %PYTHON_EXE%
%PYTHON_EXE% train_val_test.py %*
pause
