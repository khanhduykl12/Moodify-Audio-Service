@echo off
title Moodify Audio Analysis Service (Port 8002)
echo Dang khoi dong Moodify Audio Analysis Service tren cong 8002...
cd /d "%~dp0"
py -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload
if %errorlevel% neq 0 (
    python -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload
)
pause
