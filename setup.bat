:: setup.bat
:: Quick setup script for speech-domain-fix project (Windows PowerShell/CMD)
@echo off

echo [INFO] Creating virtual environment...
python -m venv venv

echo [INFO] Activating virtual environment...
call venv\Scripts\activate

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing requirements...
pip install -r requirements.txt

echo [INFO] Installing ffmpeg...
echo Please download ffmpeg manually from https://ffmpeg.org/download.html
echo After download, add ffmpeg\bin folder to your PATH environment variable.

echo [SUCCESS] Setup complete!
echo To activate the environment next time, run:
echo venv\Scripts\activate
pause
