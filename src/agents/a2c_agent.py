"""
Advantage Actor-Critic (A2C) Agent for Load Balancing

This module implements an A2C agent using PyTorch for continuous
action spaces and policy gradient methods.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for A2C agent"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor (policy) head
        self.actor_fc = nn.Linear(hidden_size, action_size)
        
        # Critic (value) head
        self.critic_fc = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor output (action probabilities)
        action_probs = F.softmax(self.actor_fc(x), dim=-1)
        
        # Critic output (state value)
        state_value = self.critic_fc(x)
        
        return action_probs, state_value


class A2CAgent:
    """
    Advantage Actor-Critic agent for load balancing
    
    This agent uses policy gradient methods with value function
    approximation, suitable for both discrete and continuous action spaces.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        hidden_size: int = 128
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network
        self.network = ActorCriticNetwork(state_size, action_size, hidden_size).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Training variables
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_loss_history': [],
            'critic_loss_history': [],
            'entropy_history': []
        }
        
    def choose_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Choose action using current policy
        
        Args:
            state: Current state as numpy array
            
        Returns:
            Tuple[int, float]: (action, log_probability)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.network(state_tensor)
            
            # Sample action from probability distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
        return action.item(), log_prob.item()
        
    def learn(self, states: List[np.ndarray], actions: List[int], 
              rewards: List[float], log_probs: List[float], done: bool):
        """
        Learn from episode using A2C
        
        Args:
            states: List of states in episode
            actions: List of actions taken
            rewards: List of rewards received
            log_probs: List of log probabilities
            done: Whether episode is done
        """
        if len(states) == 0:
            return
            
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
        
        # Calculate returns
        returns = self._calculate_returns(rewards_tensor, done)
        
        # Get current policy and value predictions
        action_probs, values = self.network(states_tensor)
        
        # Calculate advantages
        advantages = returns - values.squeeze()
        
        # Actor loss (policy gradient)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs_current = action_dist.log_prob(actions_tensor)
        actor_loss = -(log_probs_current * advantages.detach()).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy bonus for exploration
        entropy = action_dist.entropy().mean()
        
        # Total loss
        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Record losses
        self.training_history['actor_loss_history'].append(actor_loss.item())
        self.training_history['critic_loss_history'].append(critic_loss.item())
        self.training_history['entropy_history'].append(entropy.item())
        
    def _calculate_returns(self, rewards: torch.Tensor, done: bool) -> torch.Tensor:
        """Calculate discounted returns"""
        returns = []
        R = 0 if done else 0  # No bootstrap if episode is done
        
        for r in reversed(rewards):
            R = r + self.discount_factor * R
            returns.insert(0, R)
            
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
        return returns
        
    def save(self, filepath: str):
        """Save agent to file"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)
        
    def load(self, filepath: str):
        """Load agent from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
    def get_stats(self) -> Dict[str, float]:
        """Get agent statistics"""
        return {
            'avg_actor_loss': np.mean(self.training_history['actor_loss_history'][-100:]) if self.training_history['actor_loss_history'] else 0.0,
            'avg_critic_loss': np.mean(self.training_history['critic_loss_history'][-100:]) if self.training_history['critic_loss_history'] else 0.0,
            'avg_entropy': np.mean(self.training_history['entropy_history'][-100:]) if self.training_history['entropy_history'] else 0.0
        }
        
    def reset(self):
        """Reset agent to initial state"""
        self.network = ActorCriticNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_loss_history': [],
            'critic_loss_history': [],
            'entropy_history': []
        } 