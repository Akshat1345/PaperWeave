#!/bin/bash
# Quick start script for AI Research Assistant

echo "🚀 AI Research Assistant - Quick Start"
echo "======================================"
echo ""

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if Ollama is running
echo ""
echo "🔍 Checking Ollama..."
if ! ollama list > /dev/null 2>&1; then
    echo "❌ Ollama is not running!"
    echo "📝 Please run in another terminal:"
    echo "   ollama serve"
    exit 1
fi
echo "✓ Ollama is running"

# Check installed models
echo ""
echo "📦 Checking Ollama models..."
models=$(ollama list | grep -v NAME | awk '{print $1}' | head -1)
if [ -z "$models" ]; then
    echo "⚠️  No Ollama model found!"
    echo "📝 Pull a model first:"
    echo "   ollama pull llama3.2"
    echo "   or"
    echo "   ollama pull neural-chat"
    exit 1
fi
echo "✓ Model available: $models"

# Install/Update dependencies
echo ""
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt -q
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo "✓ Dependencies installed"

# Download NLTK data
echo ""
echo "📚 Downloading NLTK data..."
python -m nltk.downloader punkt averaged_perceptron_tagger -d ~/nltk_data > /dev/null 2>&1
echo "✓ NLTK data ready"

# Download spaCy model
echo ""
echo "🔤 Downloading spaCy model..."
python -m spacy download en_core_web_sm > /dev/null 2>&1
echo "✓ spaCy model ready"

# Initialize database
echo ""
echo "💾 Initializing database..."
python -c "from modules.database import db; print('✓ Database initialized')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Failed to initialize database"
    exit 1
fi

# Start application
echo ""
echo "======================================"
echo "✅ All checks passed!"
echo ""
echo "🎯 Starting AI Research Assistant..."
echo "📱 Open browser: http://localhost:5000"
echo "📝 Press Ctrl+C to stop"
echo ""
python app.py
