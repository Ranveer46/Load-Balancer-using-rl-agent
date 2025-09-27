"""
Deep Q-Network (DQN) Agent for Load Balancing

This module implements a DQN agent using PyTorch for handling
large state spaces in the load balancing environment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import Dict, List, Tuple, Optional


class DuelingDQNNetwork(nn.Module):
    """Dueling architecture: separates value and advantage streams."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super(DuelingDQNNetwork, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Value and advantage heads
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
            
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine as in dueling: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class PrioritizedReplayBuffer:
    """Lightweight proportional prioritized replay."""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = max(1, beta_frames)
        self.frame = 1
        self.eps = 1e-5
        
    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        data = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Anneal beta
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        self.frame += 1
        
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.BoolTensor(dones),
            torch.LongTensor(indices),
            weights
        )
        
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + self.eps
        
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for load balancing
    
    This agent uses a neural network to approximate Q-values,
    suitable for large state spaces.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.0005,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 250,
        hidden_size: int = 64,
        use_per: bool = True,
        use_power_of_two_choices: bool = True
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_power_of_two_choices = use_power_of_two_choices
        
        # Networks (Dueling)
        self.q_network = DuelingDQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = DuelingDQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.use_per = use_per
        if use_per:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            # Fallback to uniform replay using deque
            self.memory = deque(maxlen=memory_size)
        
        # Training variables
        self.step_count = 0
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_history': [],
            'loss_history': []
        }
        
    def choose_action(self, state: np.ndarray) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state as numpy array
            
        Returns:
            int: Chosen action
        """
        # Exploration
        if np.random.random() < self.epsilon:
            if self.use_power_of_two_choices:
                a1 = np.random.randint(0, self.action_size)
                a2 = np.random.randint(0, self.action_size)
                return a1 if np.random.random() < 0.5 else a2
            return np.random.randint(0, self.action_size)
        
        # Exploitation with power-of-two choices (subset argmax)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)
            if self.use_power_of_two_choices and self.action_size >= 2:
                candidates = torch.tensor([
                    np.random.randint(0, self.action_size),
                    np.random.randint(0, self.action_size)
                ], device=self.device)
                cand_q = q_values[candidates]
                best_idx = torch.argmax(cand_q).item()
                return int(candidates[best_idx].item())
            return int(torch.argmax(q_values).item())
                
    def learn(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool):
        """
        Learn from experience using DQN
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Store experience
        if self.use_per:
            self.memory.add(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))
        
        # Learn if enough samples
        if (len(self.memory) if self.use_per else len(self.memory)) >= self.batch_size:
            self._train()
            
        # Update target network periodically
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        self.step_count += 1
        
    def _train(self):
        """Train the Q-network using Double DQN; support PER weights."""
        if self.use_per:
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        else:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones)
            indices = None
            weights = torch.ones(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN target: action from online, value from target
        with torch.no_grad():
            next_q_online = self.q_network(next_states)
            next_actions = next_q_online.argmax(dim=1)
            next_q_target = self.target_network(next_states)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (self.discount_factor * next_q_values * (~dones))
        
        # Loss with PER weights
        td_errors = current_q_values - target_q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        # Update priorities
        if self.use_per:
            new_prios = td_errors.detach().abs().cpu().numpy() + 1e-5
            self.memory.update_priorities(indices.cpu().numpy(), new_prios)
        
        self.training_history['loss_history'].append(loss.item())
        
    def update_epsilon(self):
        """Decay epsilon for exploration"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def save(self, filepath: str):
        """Save agent to file"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_history': self.training_history
        }, filepath)
        
    def load(self, filepath: str):
        """Load agent from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_history = checkpoint['training_history']
        
    def get_stats(self) -> Dict[str, float]:
        """Get agent statistics"""
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'step_count': self.step_count,
            'avg_loss': np.mean(self.training_history['loss_history'][-100:]) if self.training_history['loss_history'] else 0.0
        }
        
    def reset(self):
        """Reset agent to initial state"""
        self.q_network = DuelingDQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = DuelingDQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        if self.use_per:
            self.memory = PrioritizedReplayBuffer()
        else:
            self.memory = deque(maxlen=10000)
        self.step_count = 0
        self.epsilon = 1.0
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_history': [],
            'loss_history': []
        } 