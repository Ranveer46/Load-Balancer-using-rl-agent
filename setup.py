#!/usr/bin/env python3
"""
Setup script for RL Load Balancer

This script installs dependencies and prepares the environment.
"""

import os
import sys
import subprocess
import platform


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "data",
        "data/models",
        "data/results",
        "data/logs",
        "data/plots"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ Created {directory}")


def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import numpy
        import torch
        import gym
        import fastapi
        import plotly
        print("✅ All core dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)


def test_environment():
    """Test the RL environment"""
    print("🔧 Testing environment...")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from environment import LoadBalancerEnv
        
        env = LoadBalancerEnv(num_servers=2, max_requests=10)
        state = env.reset()
        action = 0
        next_state, reward, done, info = env.step(action)
        
        print("✅ Environment test passed")
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        sys.exit(1)


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. 🚀 Run the demo: python demo.py")
    print("2. 📊 Start the dashboard: python start_dashboard.py")
    print("3. 🧠 Train an agent: python src/main.py --agent dqn --episodes 1000")
    print("\nQuick start:")
    print("  python demo.py")
    print("\nDashboard:")
    print("  python start_dashboard.py")
    print("  # Then open src/dashboard/frontend/index.html in your browser")
    print("\nTraining:")
    print("  python src/main.py --agent dqn --episodes 1000")
    print("\nFor more information, see README.md")


def main():
    """Main setup function"""
    print("🤖 RL Load Balancer Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    # Test imports
    test_imports()
    
    # Test environment
    test_environment()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main() 