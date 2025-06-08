#!/bin/bash

# Healthcare Monitoring System Startup Script
echo "🏥 Starting Healthcare Monitoring System..."

# Create necessary directories
mkdir -p saved_models
mkdir -p logs

# Install Python dependencies if needed
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Train the ML models if no saved models exist
if [ ! -d "saved_models" ] || [ -z "$(ls -A saved_models)" ]; then
    echo "🧠 Training ML models (this may take a few minutes)..."
    python main.py > logs/training.log 2>&1
    echo "✅ Model training completed"
else
    echo "📁 Found existing trained models"
fi

# Start the FastAPI server
echo "🚀 Starting FastAPI server..."
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
