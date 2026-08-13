@echo off
REM ============================================================
REM   OrcaMax Code — Single Entry Point
REM   Starts the local web UI and opens it in the browser
REM   If the server is already running, just opens the browser.
REM ============================================================
chcp 65001 >nul
setlocal

cd /d "D:\Hermes"

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE
set HF_HUB_DISABLE_PROGRESS_BARS=1

echo.
echo  ============================================
echo    OrcaMax Code
echo  ============================================
echo.

REM Check if a server is already running on port 7777
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 7777 -State Listen -ErrorAction SilentlyContinue; if ($c) { Write-Host 'Server already running on port 7777'; exit 0 } else { exit 1 }" >nul 2>&1

if %ERRORLEVEL%==0 (
    echo  Server already running on http://localhost:7777/
    echo  Opening browser...
    echo.
    REM Just open the browser, no need to start a new server
    powershell -NoProfile -Command ^
        "$b = $null; ^
         $paths = @(^
            'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',^
            'C:\Program Files\Microsoft\Edge\Application\msedge.exe',^
            'C:\Program Files\Google\Chrome\Application\chrome.exe',^
            'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'^
         ); ^
         foreach ($p in $paths) { if (Test-Path $p) { $b = $p; break } }; ^
         if ($b) { Start-Process $b --args '--app=http://localhost:7777/','--window-size=1200,800','--window-position=200,100' } ^
         else { Start-Process 'http://localhost:7777/' }"
    goto :eof
)

echo  Starting local server on http://localhost:7777/
echo  Browser will open automatically.
echo  Close this window to stop the server.
echo.

REM Start the browser in 3 seconds (after server warms up)
powershell -NoProfile -Command ^
    "Start-Sleep -Seconds 3; ^
     $b = $null; ^
     $paths = @(^
        'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',^
        'C:\Program Files\Microsoft\Edge\Application\msedge.exe',^
        'C:\Program Files\Google\Chrome\Application\chrome.exe',^
        'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'^
     ); ^
     foreach ($p in $paths) { if (Test-Path $p) { $b = $p; break } }; ^
     if ($b) { Start-Process $b --args '--app=http://localhost:7777/','--window-size=1200,800','--window-position=200,100' } ^
     else { Start-Process 'http://localhost:7777/' }" &

REM Run server in foreground
python webui.py

endlocal
