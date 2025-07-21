#!/usr/bin/env python3
"""
CodeConductor v2.0 Startup Script
================================

This script starts both the FastAPI server and Streamlit GUI
in separate processes, similar to the Docker deployment.

Usage:
    python start_codeconductor.py
"""

import subprocess
import sys
import time
from pathlib import Path


def print_banner():
    """Print startup banner"""
    print("🎼" + "=" * 60)
    print("🎼 CodeConductor v2.0 - Multi-Agent AI Code Generation")
    print("🎼" + "=" * 60)
    print("🚀 Starting services...")
    print("📊 API: http://localhost:8000")
    print("🎨 GUI: http://localhost:8501")
    print("📁 Data: ./data")
    print("=" * 60)


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import streamlit
        import uvicorn

        print("✅ Dependencies check passed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False


def start_api():
    """Start FastAPI server"""
    print("📊 Starting FastAPI server...")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "generated_api:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--reload",
            ]
        )
        return process
    except Exception as e:
        print(f"❌ Failed to start API: {e}")
        return None


def start_gui():
    """Start Streamlit GUI"""
    print("🎨 Starting Streamlit GUI...")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.port",
                "8501",
                "--server.headless",
                "true",
                "--server.address",
                "0.0.0.0",
            ]
        )
        return process
    except Exception as e:
        print(f"❌ Failed to start GUI: {e}")
        return None


def wait_for_services():
    """Wait for services to start"""
    print("⏳ Waiting for services to start...")
    time.sleep(5)
    print("✅ Services should be ready!")


def main():
    """Main startup function"""
    print_banner()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Create data directories
    Path("data/generated").mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    # Start services
    api_process = start_api()
    if not api_process:
        sys.exit(1)

    gui_process = start_gui()
    if not gui_process:
        api_process.terminate()
        sys.exit(1)

    # Wait for services
    wait_for_services()

    print("\n🎉 CodeConductor v2.0 is running!")
    print("📊 API: http://localhost:8000")
    print("🎨 GUI: http://localhost:8501")
    print("\nPress Ctrl+C to stop all services...")

    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
            # Check if processes are still running
            if api_process.poll() is not None:
                print("❌ API process stopped unexpectedly")
                break
            if gui_process.poll() is not None:
                print("❌ GUI process stopped unexpectedly")
                break
    except KeyboardInterrupt:
        print("\n🛑 Shutting down CodeConductor...")

        # Terminate processes
        if api_process:
            api_process.terminate()
            api_process.wait()
        if gui_process:
            gui_process.terminate()
            gui_process.wait()

        print("✅ CodeConductor stopped successfully")


if __name__ == "__main__":
    main()
