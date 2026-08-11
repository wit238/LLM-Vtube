@echo off
rem run_server.bat - start the Open-LLM-VTuber server
rem If OPENCODE_API_KEY is not set in the environment, loads it from opencode's
rem auth.json (provider "opencode-go") so the gateway-backed LLM always works.
setlocal enabledelayedexpansion
cd /d "%~dp0"

if "%OPENCODE_API_KEY%"=="" (
    echo Loading OPENCODE_API_KEY from opencode auth.json ...
    for /f "usebackq delims=" %%i in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = \"$env:USERPROFILE\.local\share\opencode\auth.json\"; try { $j = Get-Content $p -Raw | ConvertFrom-Json; [Console]::WriteLine($j.'opencode-go'.key) } catch { [Console]::WriteLine('') }"`) do (
        set "OPENCODE_API_KEY=%%i"
    )
    if "!OPENCODE_API_KEY!"=="" (
        echo WARNING: could not read OPENCODE_API_KEY from opencode auth.json
    )
)

echo Starting Open-LLM-VTuber server ...
uv run run_server.py %*
exit /b %errorlevel%