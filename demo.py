#!/usr/bin/env python3
"""
Demo script for RL Load Balancer

This script demonstrates the basic usage of the RL load balancer system.
"""

import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment import LoadBalancerEnv
from agents import DQNAgent
from utils.metrics import calculate_metrics


def demo_basic_training():
    """Demonstrate basic training of an RL agent"""
    print("🤖 RL Load Balancer Demo")
    print("=" * 50)
    
    # Create environment
    print("Creating load balancer environment...")
    env = LoadBalancerEnv(
        num_servers=3,
        max_requests=500,
        request_arrival_rate=5.0
    )
    
    # Create agent
    print("Creating DQN agent...")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Training loop
    print("Starting training...")
    num_episodes = 100
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        done = False
        while not done:
            # Choose action
            action = agent.choose_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Learn
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Update epsilon
        agent.update_epsilon()
        
        # Print progress
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Length: {episode_length}")
    
    # Get final metrics
    final_metrics = env.get_metrics()
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print(f"Average Response Time: {final_metrics.avg_response_time:.1f}ms")
    print(f"Throughput: {final_metrics.throughput:.1f} req/s")
    print(f"Load Balance Fairness: {final_metrics.load_balance_fairness:.3f}")
    print(f"Total Requests: {final_metrics.total_requests}")
    print(f"Failed Requests: {final_metrics.failed_requests}")
    
    # Show server status
    print("\nFinal Server Status:")
    for i, server in enumerate(env.servers):
        metrics = server.get_metrics()
        print(f"Server {i}: CPU={metrics.cpu_utilization:.1%}, "
              f"Memory={metrics.memory_utilization:.1%}, "
              f"Active={metrics.active_requests}, "
              f"Queue={metrics.queue_length}")


def demo_environment():
    """Demonstrate the environment functionality"""
    print("\n🔧 Environment Demo")
    print("=" * 30)
    
    # Create environment
    env = LoadBalancerEnv(num_servers=2, max_requests=100)
    
    # Run a few steps
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    for step in range(10):
        action = 0  # Always choose server 0
        next_state, reward, done, info = env.step(action)
        
        print(f"Step {step + 1}: Action={action}, Reward={reward:.2f}, Done={done}")
        
        if done:
            break
    
    # Show metrics
    metrics = env.get_metrics()
    print(f"\nFinal metrics: {metrics}")


def demo_agent_comparison():
    """Demonstrate different agent types"""
    print("\n🧠 Agent Comparison Demo")
    print("=" * 35)
    
    from agents import QLearningAgent, DQNAgent, A2CAgent
    
    env = LoadBalancerEnv(num_servers=3, max_requests=200)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agents = {
        'Q-Learning': QLearningAgent(state_size, action_size),
        'DQN': DQNAgent(state_size, action_size),
        'A2C': A2CAgent(state_size, action_size)
    }
    
    results = {}
    
    for name, agent in agents.items():
        print(f"\nTraining {name}...")
        
        # Quick training
        episode_rewards = []
        for episode in range(50):
            state = env.reset()
            episode_reward = 0
            
            done = False
            while not done:
                if name == 'A2C':
                    action, _ = agent.choose_action(state)
                else:
                    action = agent.choose_action(state)
                
                next_state, reward, done, info = env.step(action)
                
                if name == 'A2C':
                    # A2C needs episode data collection
                    pass
                else:
                    agent.learn(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            
            if hasattr(agent, 'update_epsilon'):
                agent.update_epsilon()
        
        avg_reward = sum(episode_rewards[-10:]) / 10
        results[name] = avg_reward
        print(f"{name} average reward: {avg_reward:.2f}")
    
    print(f"\nBest performing agent: {max(results, key=results.get)}")


if __name__ == "__main__":
    print("🚀 Starting RL Load Balancer Demo")
    print("This demo will show basic training, environment usage, and agent comparison.")
    print()
    
    try:
        # Run demos
        demo_environment()
        demo_basic_training()
        demo_agent_comparison()
        
        print("\n✅ Demo completed successfully!")
        print("\nTo run the full system:")
        print("1. Start the dashboard: python src/dashboard/backend/api.py")
        print("2. Open src/dashboard/frontend/index.html in your browser")
        print("3. Or run training: python src/main.py --agent dqn --episodes 1000")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt") 