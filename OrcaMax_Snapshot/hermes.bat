@echo off
REM ============================================================
REM   Hermes - Local AI Agent Launcher
REM   Starts the interactive chat with your local LLM
REM ============================================================
chcp 65001 >nul
setlocal

REM Move to the Hermes directory
cd /d "D:\Hermes"

REM Set required environment variables
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE
set HF_HUB_DISABLE_PROGRESS_BARS=1

echo.
echo  ===============================================
echo    H E R M E S   -   Local AI Agent
echo  ===============================================
echo.
echo  Loading model (first launch takes ~10-20s)...
echo.

REM Launch the chat
python cli.py chat %*

endlocal
