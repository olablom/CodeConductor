#!/bin/bash
echo "🎼 Starting CodeConductor GUI..."
echo ""
echo "Make sure you're in the virtual environment:"
source .venv/bin/activate
echo ""
echo "Starting Streamlit..."
python -m streamlit run app.py 