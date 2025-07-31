"""
Q-Learning Agent for Load Balancing

This module implements a Q-learning agent that learns to distribute
requests across servers efficiently.
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning agent for load balancing
    
    This agent uses a tabular Q-learning approach suitable for
    discrete state and action spaces.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action -> Q-value
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_history': [],
            'avg_q_values': []
        }
        
    def choose_action(self, state: np.ndarray) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state as numpy array
            
        Returns:
            int: Chosen action
        """
        # Convert state to tuple for dictionary key
        state_key = self._state_to_key(state)
        
        # Epsilon-greedy policy
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(0, self.action_size)
        else:
            # Exploitation: best action
            return np.argmax(self.q_table[state_key])
            
    def learn(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool):
        """
        Update Q-values using Q-learning update rule
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Next Q-value (max Q-value for next state)
        if done:
            next_q = 0.0
        else:
            next_q = np.max(self.q_table[next_state_key])
            
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_q - current_q
        )
        
        # Update Q-table
        self.q_table[state_key][action] = new_q
        
    def update_epsilon(self):
        """Decay epsilon for exploration"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def _state_to_key(self, state: np.ndarray) -> Tuple:
        """Convert state array to tuple for dictionary key"""
        # Discretize state for tabular Q-learning
        # This is a simple discretization - you might want to use more sophisticated methods
        discretized = []
        for i, value in enumerate(state):
            # Discretize into 10 bins
            bin_idx = min(9, int(value * 10))
            discretized.append(bin_idx)
        return tuple(discretized)
        
    def save(self, filepath: str):
        """Save agent to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'training_history': self.training_history
            }, f)
            
    def load(self, filepath: str):
        """Load agent from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.action_size))
            self.q_table.update(data['q_table'])
            self.epsilon = data['epsilon']
            self.training_history = data['training_history']
            
    def get_stats(self) -> Dict[str, float]:
        """Get agent statistics"""
        # Calculate average Q-values
        all_q_values = []
        for state_q_values in self.q_table.values():
            all_q_values.extend(state_q_values)
            
        avg_q = np.mean(all_q_values) if all_q_values else 0.0
        
        return {
            'epsilon': self.epsilon,
            'num_states': len(self.q_table),
            'avg_q_value': avg_q,
            'max_q_value': np.max(all_q_values) if all_q_values else 0.0,
            'min_q_value': np.min(all_q_values) if all_q_values else 0.0
        }
        
    def reset(self):
        """Reset agent to initial state"""
        self.q_table.clear()
        self.epsilon = 0.1
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_history': [],
            'avg_q_values': []
        } 