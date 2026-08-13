@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
set PATH=C:\Users\Yahia\.cargo\bin;%PATH%
set CC=cl.exe
set CXX=cl.exe
set CC_x86_64_pc_windows_msvc=cl.exe
set CXX_x86_64_pc_windows_msvc=cl.exe
set CC_x86_64-pc-windows-msvc=cl.exe
set CXX_x86_64-pc-windows-msvc=cl.exe
cd /d D:\Hermes\imported_sources\Probe
cargo build --release --offline
if errorlevel 1 exit /b %errorlevel%
if not exist D:\Hermes\bin mkdir D:\Hermes\bin
copy /y target\release\probe.exe D:\Hermes\bin\probe.exe
exit /b %errorlevel%
