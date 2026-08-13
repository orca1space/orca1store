@echo off
setlocal
cd /d D:\Hermes
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE
set HF_HUB_DISABLE_PROGRESS_BARS=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
set HERMES_OFFLINE_ONLY=1
if not exist logs mkdir logs
set PYTHON_EXE=C:\Users\Yahia\AppData\Local\Programs\Python\Python314\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python
if not exist "%PYTHON_EXE%" (
  echo Python interpreter not found.
  exit /b 1
)
echo Starting OrcaMax Code local-only server on http://127.0.0.1:7777/
"%PYTHON_EXE%" webui.py
set EXIT_CODE=%ERRORLEVEL%
echo OrcaMax exited with code %EXIT_CODE% at %DATE% %TIME%>> logs\launcher.log
endlocal & exit /b %EXIT_CODE%
