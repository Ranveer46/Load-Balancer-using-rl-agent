"""
Main training script for RL Load Balancer

This script demonstrates how to train different RL agents
for load balancing and compare their performance.
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, List, Any

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import LoadBalancerEnv, LoadBalancingAlgorithm
from agents import QLearningAgent, DQNAgent, A2CAgent
from utils.metrics import calculate_metrics, compare_algorithms
from utils.visualization import plot_training_progress, plot_comparison


def train_agent(
    agent_type: str,
    num_episodes: int = 1000,
    num_servers: int = 5,
    max_requests: int = 1000,
    request_arrival_rate: float = 10.0,
    learning_rate: float = 0.001,
    discount_factor: float = 0.95,
    epsilon: float = 0.1,
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Train an RL agent for load balancing
    
    Args:
        agent_type: Type of agent ('q_learning', 'dqn', 'a2c')
        num_episodes: Number of training episodes
        num_servers: Number of servers in the environment
        max_requests: Maximum requests per episode
        request_arrival_rate: Request arrival rate
        learning_rate: Learning rate for the agent
        discount_factor: Discount factor for RL
        epsilon: Initial exploration rate
        save_model: Whether to save the trained model
        
    Returns:
        Dictionary with training results
    """
    print(f"Training {agent_type.upper()} agent...")
    
    # Create environment
    env = LoadBalancerEnv(
        num_servers=num_servers,
        max_requests=max_requests,
        request_arrival_rate=request_arrival_rate
    )
    
    # Create agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    if agent_type == "q_learning":
        agent = QLearningAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon
        )
    elif agent_type == "dqn":
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon
        )
    elif agent_type == "a2c":
        agent = A2CAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Training variables
    episode_rewards = []
    episode_lengths = []
    avg_response_times = []
    throughputs = []
    fairness_scores = []
    
    # Training loop
    start_time = time.time()
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # For A2C, collect episode data
        if agent_type == "a2c":
            states, actions, rewards, log_probs = [], [], [], []
        
        done = False
        while not done:
            # Choose action
            if agent_type == "a2c":
                action, log_prob = agent.choose_action(state)
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
            else:
                action = agent.choose_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Learn
            if agent_type == "a2c":
                rewards.append(reward)
            else:
                agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Learn for A2C (episode-based)
        if agent_type == "a2c":
            agent.learn(states, actions, rewards, log_probs, done)
        
        # Update agent epsilon
        if hasattr(agent, 'update_epsilon'):
            agent.update_epsilon()
        
        # Record episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Get final metrics
        final_metrics = env.get_metrics()
        avg_response_times.append(final_metrics.avg_response_time)
        throughputs.append(final_metrics.throughput)
        fairness_scores.append(final_metrics.load_balance_fairness)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / 100
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Response Time: {final_metrics.avg_response_time:.1f}ms, "
                  f"Throughput: {final_metrics.throughput:.1f} req/s")
    
    training_time = time.time() - start_time
    
    # Save model
    if save_model:
        os.makedirs('data/models', exist_ok=True)
        model_path = f'data/models/{agent_type}_agent.pth'
        agent.save(model_path)
        print(f"Model saved to {model_path}")
    
    # Compile results
    results = {
        'agent_type': agent_type,
        'num_episodes': num_episodes,
        'training_time': training_time,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avg_response_times': avg_response_times,
        'throughputs': throughputs,
        'fairness_scores': fairness_scores,
        'final_metrics': {
            'avg_response_time': final_metrics.avg_response_time,
            'throughput': final_metrics.throughput,
            'fairness': final_metrics.load_balance_fairness,
            'total_requests': final_metrics.total_requests,
            'failed_requests': final_metrics.failed_requests
        },
        'agent_stats': agent.get_stats()
    }
    
    return results


def compare_with_traditional_algorithms(env: LoadBalancerEnv) -> Dict[str, Any]:
    """
    Compare RL performance with traditional algorithms
    
    Args:
        env: Trained environment
        
    Returns:
        Dictionary with comparison results
    """
    print("Comparing with traditional algorithms...")
    
    # Get metrics for traditional algorithms
    traditional_metrics = {}
    
    for algo in LoadBalancingAlgorithm:
        if algo != LoadBalancingAlgorithm.RL:
            metrics = env.get_traditional_algorithm_metrics(algo)
            traditional_metrics[algo.value] = metrics
    
    return traditional_metrics


def save_results(results: Dict[str, Any], output_dir: str = "data/results"):
    """
    Save training results to files
    
    Args:
        results: Training results dictionary
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results as JSON
    results_file = os.path.join(output_dir, f"{results['agent_type']}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Training progress plot
    progress_plot = plot_training_progress(
        episode_rewards=results['episode_rewards'],
        episode_lengths=results['episode_lengths'],
        avg_response_times=results['avg_response_times'],
        throughputs=results['throughputs'],
        save_path=os.path.join(plots_dir, f"{results['agent_type']}_progress.html")
    )
    
    print(f"Results saved to {output_dir}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train RL Load Balancer")
    parser.add_argument("--agent", type=str, default="dqn",
                       choices=["q_learning", "dqn", "a2c"],
                       help="Type of RL agent")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes")
    parser.add_argument("--servers", type=int, default=5,
                       help="Number of servers")
    parser.add_argument("--requests", type=int, default=1000,
                       help="Maximum requests per episode")
    parser.add_argument("--arrival-rate", type=float, default=10.0,
                       help="Request arrival rate")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--discount-factor", type=float, default=0.95,
                       help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1,
                       help="Initial exploration rate")
    parser.add_argument("--output-dir", type=str, default="data/results",
                       help="Output directory for results")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save the trained model")
    
    args = parser.parse_args()
    
    # Train agent
    results = train_agent(
        agent_type=args.agent,
        num_episodes=args.episodes,
        num_servers=args.servers,
        max_requests=args.requests,
        request_arrival_rate=args.arrival_rate,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        save_model=not args.no_save
    )
    
    # Save results
    save_results(results, args.output_dir)
    
    # Print final results
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Agent Type: {results['agent_type'].upper()}")
    print(f"Training Time: {results['training_time']:.2f} seconds")
    print(f"Final Average Reward: {sum(results['episode_rewards'][-100:]) / 100:.2f}")
    print(f"Final Response Time: {results['final_metrics']['avg_response_time']:.1f}ms")
    print(f"Final Throughput: {results['final_metrics']['throughput']:.1f} req/s")
    print(f"Final Fairness: {results['final_metrics']['fairness']:.3f}")
    print(f"Total Requests: {results['final_metrics']['total_requests']}")
    print(f"Failed Requests: {results['final_metrics']['failed_requests']}")
    print("="*50)


if __name__ == "__main__":
    main() 