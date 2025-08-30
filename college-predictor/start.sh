#!/bin/bash

# College Predictor Quick Start Script

set -e

echo "🎓 College Predictor - Quick Start"
echo "=================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install -r backend/requirements.txt

# Train model if it doesn't exist
if [ ! -f "models/college_predictor_model.pkl" ]; then
    echo "🤖 Training ML model..."
    cd backend
    python train_model.py
    cd ..
fi

echo "✅ Setup complete!"
echo ""
echo "🚀 Starting services..."
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"

# Start backend in background
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend
cd frontend
python -m http.server 3000 &
FRONTEND_PID=$!
cd ..

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "✅ Services stopped"
    exit 0
}

# Set up trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for user to stop
wait
