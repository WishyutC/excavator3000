@echo off
setlocal
cd /d "%~dp0"
title Excavator3000 Training Dashboard
python dashboard.py
if errorlevel 1 (
  echo.
  echo Dashboard stopped with an error.
  pause
)
