"""
FastAPI Backend for RL Load Balancer Dashboard

This module provides REST API endpoints for the dashboard to display
real-time metrics and training progress.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
import time
from datetime import datetime

# Import RL components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from environment import LoadBalancerEnv, LoadBalancingAlgorithm
from agents import QLearningAgent, DQNAgent, A2CAgent


# Global variables for storing training state
training_state = {
    'is_training': False,
    'current_episode': 0,
    'total_episodes': 0,
    'current_metrics': {},
    'episode_history': [],
    'agent_stats': {},
    'environment': None,
    'agent': None
}


class TrainingConfig(BaseModel):
    """Configuration for training"""
    agent_type: str = "dqn"  # "q_learning", "dqn", "a2c"
    num_episodes: int = 1000
    num_servers: int = 5
    max_requests: int = 1000
    request_arrival_rate: float = 10.0
    learning_rate: float = 0.001
    discount_factor: float = 0.95
    epsilon: float = 0.1


class MetricsResponse(BaseModel):
    """Response model for metrics"""
    current_episode: int
    total_episodes: int
    is_training: bool
    metrics: Dict[str, Any]
    agent_stats: Dict[str, Any]
    episode_history: List[Dict[str, Any]]


# Create FastAPI app
app = FastAPI(
    title="RL Load Balancer Dashboard",
    description="Real-time dashboard for monitoring RL load balancer training",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "RL Load Balancer Dashboard API"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "training_state": training_state['is_training']
    }


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get current training metrics"""
    return MetricsResponse(
        current_episode=training_state['current_episode'],
        total_episodes=training_state['total_episodes'],
        is_training=training_state['is_training'],
        metrics=training_state['current_metrics'],
        agent_stats=training_state['agent_stats'],
        episode_history=training_state['episode_history']
    )


@app.post("/start_training")
async def start_training(config: TrainingConfig):
    """Start RL training"""
    if training_state['is_training']:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    try:
        # Initialize environment
        env = LoadBalancerEnv(
            num_servers=config.num_servers,
            max_requests=config.max_requests,
            request_arrival_rate=config.request_arrival_rate
        )
        
        # Initialize agent
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        if config.agent_type == "q_learning":
            agent = QLearningAgent(state_size, action_size)
        elif config.agent_type == "dqn":
            agent = DQNAgent(state_size, action_size, learning_rate=config.learning_rate)
        elif config.agent_type == "a2c":
            agent = A2CAgent(state_size, action_size, learning_rate=config.learning_rate)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown agent type: {config.agent_type}")
        
        # Update training state
        training_state.update({
            'is_training': True,
            'current_episode': 0,
            'total_episodes': config.num_episodes,
            'environment': env,
            'agent': agent,
            'episode_history': [],
            'current_metrics': {},
            'agent_stats': {}
        })
        
        # Start training in background
        asyncio.create_task(train_agent(config))
        
        return {"message": "Training started", "config": config.dict()}
        
    except Exception as e:
        training_state['is_training'] = False
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@app.post("/stop_training")
async def stop_training():
    """Stop RL training"""
    training_state['is_training'] = False
    return {"message": "Training stopped"}


@app.get("/traditional_algorithms")
async def get_traditional_algorithms():
    """Get metrics for traditional load balancing algorithms"""
    if not training_state['environment']:
        return {"error": "No environment available"}
        
    env = training_state['environment']
    
    algorithms = {}
    for algo in LoadBalancingAlgorithm:
        if algo != LoadBalancingAlgorithm.RL:
            metrics = env.get_traditional_algorithm_metrics(algo)
            algorithms[algo.value] = metrics
            
    return algorithms


@app.get("/server_status")
async def get_server_status():
    """Get current server status"""
    if not training_state['environment']:
        return {"error": "No environment available"}
        
    env = training_state['environment']
    server_metrics = []
    
    for i, server in enumerate(env.servers):
        metrics = server.get_metrics()
        server_metrics.append({
            'server_id': i,
            'cpu_utilization': metrics.cpu_utilization,
            'memory_utilization': metrics.memory_utilization,
            'active_requests': metrics.active_requests,
            'queue_length': metrics.queue_length,
            'avg_response_time': metrics.avg_response_time,
            'status': metrics.status.value,
            'total_requests': metrics.total_requests,
            'failed_requests': metrics.failed_requests
        })
        
    return {"servers": server_metrics}


async def train_agent(config: TrainingConfig):
    """Train the RL agent"""
    env = training_state['environment']
    agent = training_state['agent']
    
    episode_rewards = []
    
    for episode in range(config.num_episodes):
        if not training_state['is_training']:
            break
            
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # For A2C, we need to collect episode data
        if config.agent_type == "a2c":
            states, actions, rewards, log_probs = [], [], [], []
        
        done = False
        while not done:
            # Choose action
            if config.agent_type == "a2c":
                action, log_prob = agent.choose_action(state)
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
            else:
                action = agent.choose_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Learn
            if config.agent_type == "a2c":
                rewards.append(reward)
            else:
                agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Update metrics for dashboard
            training_state['current_metrics'] = {
                'total_requests': info['total_requests'],
                'completed_requests': info['completed_requests'],
                'failed_requests': info['failed_requests'],
                'avg_response_time': info['avg_response_time'],
                'throughput': info['throughput'],
                'load_balance_fairness': info['load_balance_fairness'],
                'server_utilizations': info['server_utilizations']
            }
        
        # Learn for A2C (episode-based)
        if config.agent_type == "a2c":
            agent.learn(states, actions, rewards, log_probs, done)
        
        # Update agent epsilon
        if hasattr(agent, 'update_epsilon'):
            agent.update_epsilon()
        
        # Record episode results
        episode_rewards.append(episode_reward)
        
        # Update training state
        training_state['current_episode'] = episode + 1
        training_state['agent_stats'] = agent.get_stats()
        
        # Add to episode history (keep last 100)
        episode_data = {
            'episode': episode + 1,
            'reward': episode_reward,
            'length': episode_length,
            'avg_response_time': training_state['current_metrics']['avg_response_time'],
            'throughput': training_state['current_metrics']['throughput'],
            'fairness': training_state['current_metrics']['load_balance_fairness']
        }
        
        training_state['episode_history'].append(episode_data)
        if len(training_state['episode_history']) > 100:
            training_state['episode_history'] = training_state['episode_history'][-100:]
        
        # Small delay to allow dashboard updates
        await asyncio.sleep(0.01)
    
    # Training completed
    training_state['is_training'] = False
    
    # Save trained agent
    if agent:
        os.makedirs('data/models', exist_ok=True)
        agent.save(f'data/models/{config.agent_type}_agent.pth')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 