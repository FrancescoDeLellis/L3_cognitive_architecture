@echo off
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%"

echo Stopping servers...

REM ---- Kill server 60000/60001 ----
if exist pid_60000.txt (
    powershell -NoProfile -Command "Get-Content pid_60000.txt | ForEach-Object { Stop-Process -Id $_ -Force }"
    del pid_60000.txt
    echo Stopped 60000/60001
)

REM ---- Kill server 60010/60011 ----
if exist pid_60010.txt (
    powershell -NoProfile -Command "Get-Content pid_60010.txt | ForEach-Object { Stop-Process -Id $_ -Force }"
    del pid_60010.txt
    echo Stopped 60010/60011
)

REM ---- Kill server 60020/60021 ----
if exist pid_60020.txt (
    powershell -NoProfile -Command "Get-Content pid_60020.txt | ForEach-Object { Stop-Process -Id $_ -Force }"
    del pid_60020.txt
    echo Stopped 60020/60021
)

popd
