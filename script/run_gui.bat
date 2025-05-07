@echo off
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Launching GUI...
python Gui.py

pause
