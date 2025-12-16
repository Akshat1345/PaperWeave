@echo off
REM Quick start script for AI Research Assistant (Windows)

echo.
echo 🚀 AI Research Assistant - Quick Start
echo =====================================
echo.

REM Check Python version
python --version
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.10+
    exit /b 1
)

REM Check Ollama
echo.
echo 🔍 Checking Ollama...
ollama list >nul 2>&1
if errorlevel 1 (
    echo ❌ Ollama is not running!
    echo 📝 Please run Ollama first:
    echo    ollama serve
    exit /b 1
)
echo ✓ Ollama is running

REM Install dependencies
echo.
echo 📥 Installing Python dependencies...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    exit /b 1
)
echo ✓ Dependencies installed

REM Download NLTK data
echo.
echo 📚 Downloading NLTK data...
python -m nltk.downloader punkt averaged_perceptron_tagger >nul 2>&1
echo ✓ NLTK data ready

REM Download spaCy model
echo.
echo 🔤 Downloading spaCy model...
python -m spacy download en_core_web_sm >nul 2>&1
echo ✓ spaCy model ready

REM Initialize database
echo.
echo 💾 Initializing database...
python -c "from modules.database import db; print('✓ Database initialized')" 2>nul
if errorlevel 1 (
    echo ❌ Failed to initialize database
    exit /b 1
)

REM Start application
echo.
echo =====================================
echo ✅ All checks passed!
echo.
echo 🎯 Starting AI Research Assistant...
echo 📱 Open browser: http://localhost:5000
echo 📝 Press Ctrl+C to stop
echo.
python app.py
