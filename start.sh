#!/bin/bash
# Production startup script for AI Research Assistant

set -e

echo "🚀 AI Research Assistant - Production Startup"
echo "=============================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your production settings!"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check Ollama
echo "🤖 Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found! Please install from https://ollama.ai"
    exit 1
fi

# Check if model is available
echo "📥 Checking Ollama model..."
if ! ollama list | grep -q "${OLLAMA_MODEL:-llama3.2:latest}"; then
    echo "⚠️  Model not found. Pulling ${OLLAMA_MODEL:-llama3.2:latest}..."
    ollama pull "${OLLAMA_MODEL:-llama3.2:latest}"
fi

# Create required directories
echo "📁 Creating directories..."
mkdir -p data/pdfs processed/{cache,compiled,chroma_db,images,graph_exports} logs static/css

# Check dependencies
echo "📚 Checking dependencies..."
pip list | grep -q Flask || {
    echo "❌ Dependencies not installed!"
    echo "Run: pip install -r requirements.txt"
    exit 1
}

# Get configuration
WORKERS=${WORKERS:-4}
PORT=${PORT:-5000}
HOST=${HOST:-0.0.0.0}
FLASK_ENV=${FLASK_ENV:-production}

echo ""
echo "✅ All checks passed!"
echo ""
echo "Configuration:"
echo "  Environment: $FLASK_ENV"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "  Ollama Model: ${OLLAMA_MODEL:-llama3.2:latest}"
echo ""

# Start server based on environment
if [ "$FLASK_ENV" = "production" ]; then
    echo "🚀 Starting Gunicorn (Production Mode)..."
    exec gunicorn \
        --workers $WORKERS \
        --bind $HOST:$PORT \
        --timeout 180 \
        --access-logfile logs/access.log \
        --error-logfile logs/error.log \
        --log-level info \
        app:app
else
    echo "🔧 Starting Flask Development Server..."
    exec python app.py
fi
