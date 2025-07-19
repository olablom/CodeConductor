#!/bin/bash

echo "🎼 Starting CodeConductor GUI..."
echo "=================================="

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not detected. Please activate it first:"
    echo "source .venv/bin/activate  # Linux/Mac"
    echo ".venv\\Scripts\\activate     # Windows"
    exit 1
fi

# Check if Streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing Streamlit..."
    pip install streamlit
fi

# Create data directory if it doesn't exist
mkdir -p data

echo "🚀 Starting CodeConductor Dashboard..."
echo "🌐 Opening http://localhost:8501"
echo ""
echo "🎯 Features:"
echo "  - Multi-Agent Discussion"
echo "  - Human-in-the-Loop Approval"
echo "  - RL Learning Metrics"
echo "  - Generated Code Preview"
echo "  - Project History"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Streamlit
streamlit run app.py --server.port 8501 --server.address localhost 