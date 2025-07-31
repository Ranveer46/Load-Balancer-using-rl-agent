#!/usr/bin/env python3
"""
Startup script for RL Load Balancer Dashboard

This script starts the FastAPI backend server for the dashboard.
"""

import os
import sys
import uvicorn
import webbrowser
import time
import threading

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dashboard.backend.api import app


def open_dashboard():
    """Open the dashboard in the default browser"""
    time.sleep(3)  # Wait for server to start
    dashboard_url = "file://" + os.path.abspath("src/dashboard/frontend/index.html")
    webbrowser.open(dashboard_url)
    print(f"🌐 Dashboard opened at: {dashboard_url}")


def main():
    """Start the dashboard server"""
    print("🚀 Starting RL Load Balancer Dashboard")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)
    
    print("📁 Created data directories")
    print("🔧 Starting FastAPI server...")
    
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_dashboard)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the server
    uvicorn.run(
        "src.dashboard.backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main() 