@echo off

@REM Simple start
@REM set SCRIPT_DIR=%~dp0
@REM pushd "%SCRIPT_DIR%"

@REM start "" cmd /k python L3_server_dual_socket.py 60000 60001
@REM start "CA2" cmd /k python3 L3_server_dual_socket.py 60010 60011
@REM start "CA3" cmd /k python3 L3_server_dual_socket.py 60020 60021


@REM PowerShell start
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%"

echo Starting servers...

set OutIp=127.0.0.1
@REM set OutIp=192.168.101.223
echo --------------------------------
echo %OutIp%

set COUNT=%1
if "%COUNT%"=="" set COUNT=3

REM ---- Server 1 ----
powershell -NoProfile -Command ^
 "$p = Start-Process 'python.exe' -ArgumentList 'L3_server_dual_socket.py 60000 60001 %OutIp%' -WorkingDirectory '%SCRIPT_DIR%' -PassThru -WindowStyle Normal; $p.Id | Out-File 'pid_60000.txt' -Encoding ascii"
echo Server 60000/60001 started.

if %COUNT% GTR 1 (
    REM ---- Server 2 ----
    powershell -NoProfile -Command ^
     "$p = Start-Process 'python.exe' -ArgumentList 'L3_server_dual_socket.py 60010 60011 %OutIp%' -WorkingDirectory '%SCRIPT_DIR%' -PassThru -WindowStyle Normal; $p.Id | Out-File 'pid_60010.txt' -Encoding ascii"
    echo Server 60010/60011 started.
)

if %COUNT% GTR 2 (
    REM ---- Server 3 ----
    powershell -NoProfile -Command ^
     "$p = Start-Process 'python.exe' -ArgumentList 'L3_server_dual_socket.py 60020 60021 %OutIp%' -WorkingDirectory '%SCRIPT_DIR%' -PassThru -WindowStyle Normal; $p.Id | Out-File 'pid_60020.txt' -Encoding ascii"
    echo Server 60020/60021 started.
)
popd