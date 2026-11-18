@echo off
cd /d "%~dp0"
if exist "..\.venv-stock\Scripts\python.exe" (
    "..\.venv-stock\Scripts\python.exe" -X utf8 key_codec.py
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -X utf8 key_codec.py
) else (
    py -3 -X utf8 key_codec.py
)
pause
