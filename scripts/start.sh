#!/bin/bash

# Garment Creative AI - Start Script

echo "🚀 Starting Garment Creative AI..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p app/static/uploads
mkdir -p app/static/outputs
mkdir -p logs

# Check if Redis is running
echo "🔍 Checking Redis connection..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Please start Redis first:"
    echo "   Ubuntu/Debian: sudo systemctl start redis-server"
    echo "   macOS: brew services start redis"
    echo "   Docker: docker run -d -p 6379:6379 redis:alpine"
    exit 1
fi

echo "✅ Redis is running"

# Install dependencies if needed
if [ ! -d "venv" ] && [ ! -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Start Celery worker in background
echo "🔧 Starting Celery worker..."
celery -A app.services.celery_app worker --loglevel=info --detach --pidfile=celery_worker.pid

# Start Celery beat in background
echo "⏰ Starting Celery beat..."
celery -A app.services.celery_app beat --loglevel=info --detach --pidfile=celery_beat.pid

# Wait a moment for Celery to start
sleep 2

# Start FastAPI application
echo "🌐 Starting FastAPI application..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Cleanup function
cleanup() {
    echo "🛑 Shutting down services..."
    if [ -f celery_worker.pid ]; then
        kill $(cat celery_worker.pid) 2>/dev/null
        rm celery_worker.pid
    fi
    if [ -f celery_beat.pid ]; then
        kill $(cat celery_beat.pid) 2>/dev/null
        rm celery_beat.pid
    fi
    echo "✅ Cleanup completed"
}

# Set trap to cleanup on script exit
trap cleanup EXIT

# Wait for user interrupt
wait