#!/bin/bash
# setup.sh
# Quick setup script for speech-domain-fix project (Linux/Mac/WSL)

echo "[INFO] Creating virtual environment..."
python3 -m venv venv

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

echo "[INFO] Upgrading pip..."
pip install --upgrade pip

echo "[INFO] Installing project dependencies..."
pip install -r requirements.txt

echo "[INFO] Installing system dependencies (ffmpeg)..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y ffmpeg
elif command -v brew &> /dev/null; then
    brew install ffmpeg
else
    echo "[WARNING] Could not detect package manager. Please install ffmpeg manually."
fi

echo "[SUCCESS] Setup complete! Activate venv with:"
echo "source venv/bin/activate"
