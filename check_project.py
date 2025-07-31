#!/usr/bin/env python3
"""
Project Status Checker

This script checks if all components of the RL Load Balancer project are working.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_imports():
    """Check if all modules can be imported"""
    print("🔍 Checking imports...")
    
    try:
        from environment import LoadBalancerEnv
        print("✅ Environment module imported")
        
        from agents import DQNAgent, QLearningAgent, A2CAgent
        print("✅ Agent modules imported")
        
        from utils.metrics import calculate_metrics
        print("✅ Utils modules imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def check_environment():
    """Check if environment works"""
    print("\n🔧 Testing environment...")
    
    try:
        from environment import LoadBalancerEnv
        
        env = LoadBalancerEnv(num_servers=2, max_requests=10)
        state = env.reset()
        action = 0
        next_state, reward, done, info = env.step(action)
        
        print(f"✅ Environment test passed")
        print(f"   State shape: {state.shape}")
        print(f"   Action space: {env.action_space.n}")
        print(f"   Reward: {reward}")
        
        return True
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def check_agents():
    """Check if agents work"""
    print("\n🧠 Testing agents...")
    
    try:
        from environment import LoadBalancerEnv
        from agents import DQNAgent, QLearningAgent, A2CAgent
        
        env = LoadBalancerEnv(num_servers=2, max_requests=10)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        # Test DQN
        dqn = DQNAgent(state_size, action_size)
        action = dqn.choose_action(env.reset())
        print(f"✅ DQN agent working (action: {action})")
        
        # Test Q-Learning
        ql = QLearningAgent(state_size, action_size)
        action = ql.choose_action(env.reset())
        print(f"✅ Q-Learning agent working (action: {action})")
        
        # Test A2C
        a2c = A2CAgent(state_size, action_size)
        action, _ = a2c.choose_action(env.reset())
        print(f"✅ A2C agent working (action: {action})")
        
        return True
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def check_dashboard():
    """Check if dashboard components work"""
    print("\n📊 Testing dashboard components...")
    
    try:
        import fastapi
        print("✅ FastAPI imported")
        
        import plotly
        print("✅ Plotly imported")
        
        # Check if dashboard files exist
        dashboard_files = [
            "src/dashboard/frontend/index.html",
            "src/dashboard/frontend/style.css", 
            "src/dashboard/frontend/script.js",
            "src/dashboard/backend/api.py"
        ]
        
        for file in dashboard_files:
            if os.path.exists(file):
                print(f"✅ {file} exists")
            else:
                print(f"❌ {file} missing")
        
        return True
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def main():
    """Run all checks"""
    print("🤖 RL Load Balancer Project Status Check")
    print("=" * 50)
    
    checks = [
        check_imports,
        check_environment,
        check_agents,
        check_dashboard
    ]
    
    passed = 0
    total = len(checks)
    
    for check in checks:
        if check():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All systems operational!")
        print("\nNext steps:")
        print("1. Run demo: python demo.py")
        print("2. Start dashboard: python start_dashboard.py")
        print("3. Train agent: python src/main.py --agent dqn --episodes 1000")
    else:
        print("⚠️ Some issues detected. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 