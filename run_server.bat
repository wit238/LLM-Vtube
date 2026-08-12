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

rem If DASHSCOPE_API_KEY is not set, loads it from dashscope_key.txt (one line,
rem gitignored) so Qwen3-TTS works without pasting the key into conf.yaml.
if "%DASHSCOPE_API_KEY%"=="" (
    if exist "dashscope_key.txt" (
        set /p DASHSCOPE_API_KEY=<dashscope_key.txt
        echo Loaded DASHSCOPE_API_KEY from dashscope_key.txt
    )
)


echo Starting Open-LLM-VTuber server ...
uv run run_server.py %*
set "EXIT_CODE=%errorlevel%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Server exited with code %EXIT_CODE%. See the messages above.
    pause
)
exit /b %EXIT_CODE%
