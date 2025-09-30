#!/bin/bash

# RAG FUSSA Development Startup Script

echo "🚀 Starting RAG FUSSA Development Environment"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run: python -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if Redis is running
echo "🔍 Checking Redis connection..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo "⚠️  Redis is not running. Please start Redis first:"
    echo "   docker run -d -p 6379:6379 redis:7-alpine"
    echo "   OR"
    echo "   brew services start redis"
    exit 1
fi

echo "✅ Redis is running"

# Start Celery worker in background
echo "🔄 Starting Celery worker..."
python start_worker.py &
CELERY_PID=$!

# Wait a moment for worker to start
sleep 3

# Check if worker is running
if ! kill -0 $CELERY_PID 2>/dev/null; then
    echo "❌ Failed to start Celery worker"
    exit 1
fi

echo "✅ Celery worker started (PID: $CELERY_PID)"

# Start FastAPI server
echo "🌐 Starting FastAPI server..."
echo "   API will be available at: http://localhost:8000"
echo "   API docs at: http://localhost:8000/docs"
echo "   Health check at: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $CELERY_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup INT TERM

# Start FastAPI server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
