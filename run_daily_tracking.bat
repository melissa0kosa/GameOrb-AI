@echo off
REM Daily pre-launch tracking run (registered in Windows Task Scheduler).
REM Appends output to data\tracking.log.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo ===== %DATE% %TIME% ===== >> data\tracking.log
".venv\Scripts\python.exe" track_upcoming.py >> data\tracking.log 2>&1
