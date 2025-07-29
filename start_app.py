#!/usr/bin/env python3
"""
Startup script for CodeConductor MVP
Uses streamlit run to avoid ScriptRunContext warnings
"""

import os
import sys
import subprocess

# Set environment variables to suppress warnings
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_LOGGER_LEVEL"] = "error"

print("🚀 Starting CodeConductor MVP...")
print("✅ Environment variables set to suppress warnings")

# Run the app using streamlit run
try:
    subprocess.run(
        [
            "streamlit",
            "run",
            "codeconductor_app.py",
            "--server.headless",
            "true",
            "--server.runOnSave",
            "false",
            "--logger.level",
            "error",
        ],
        check=True,
    )
except KeyboardInterrupt:
    print("\n👋 App stopped by user")
except Exception as e:
    print(f"❌ Error starting app: {e}")
