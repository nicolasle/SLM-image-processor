@echo off
title SLM Image Processor
cd /d "%~dp0"
"C:\Program Files\Python314\python.exe" main.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
