@echo off
REM Run from project root so that "from backend.X import Y" works.
cd /d "%~dp0"
call venv\Scripts\activate.bat
set PYTHONPATH=%~dp0
uvicorn backend.api:app --reload
