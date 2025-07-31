"""
Load Balancer Environment for Reinforcement Learning

This module implements a Gym-compatible environment for training RL agents
to perform load balancing across multiple servers.
"""

import gymnasium as gym
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .server_simulator import ServerSimulator, ServerStatus


class LoadBalancingAlgorithm(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    RL = "rl"


@dataclass
class EnvironmentMetrics:
    """Overall environment performance metrics"""
    total_requests: int
    completed_requests: int
    failed_requests: int
    avg_response_time: float
    throughput: float  # requests per second
    load_balance_fairness: float  # Jain's fairness index
    server_utilizations: List[float]


class LoadBalancerEnv(gym.Env):
    """
    Reinforcement Learning Environment for Load Balancing
    
    This environment simulates a load balancer that distributes incoming
    web requests across multiple servers. The RL agent learns to choose
    which server to assign each request to minimize response time and
    maximize throughput.
    """
    
    def __init__(
        self,
        num_servers: int = 5,
        max_requests: int = 1000,
        request_arrival_rate: float = 10.0,  # requests per second
        request_size_distribution: str = "exponential",  # exponential, uniform, normal
        enable_server_failures: bool = True,
        enable_traffic_spikes: bool = True
    ):
        super().__init__()
        
        self.num_servers = num_servers
        self.max_requests = max_requests
        self.request_arrival_rate = request_arrival_rate
        self.request_size_distribution = request_size_distribution
        self.enable_server_failures = enable_server_failures
        self.enable_traffic_spikes = enable_traffic_spikes
        
        # Initialize servers
        self.servers = []
        for i in range(num_servers):
            server = ServerSimulator(
                server_id=i,
                failure_probability=0.001 if enable_server_failures else 0.0
            )
            self.servers.append(server)
            
        # Environment state
        self.current_step = 0
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.current_time = 0.0
        self.last_request_time = 0.0
        
        # Request tracking
        self.pending_requests = []
        self.completed_requests_history = []
        self.request_id_counter = 0
        
        # Performance tracking
        self.response_times = []
        self.server_utilizations_history = []
        self.reward_history = []
        
        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(num_servers)
        
        # State space: server metrics + environment metrics
        state_size = (
            num_servers * 8 +  # 8 metrics per server
            5  # 5 global metrics
        )
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(state_size,), dtype=np.float32
        )
        
        # Traditional algorithms for comparison
        self.round_robin_counter = 0
        self.least_connections_history = []
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.current_step = 0
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.current_time = 0.0
        self.last_request_time = 0.0
        
        # Reset servers
        for server in self.servers:
            server.reset()
            
        # Clear history
        self.pending_requests.clear()
        self.completed_requests_history.clear()
        self.response_times.clear()
        self.server_utilizations_history.clear()
        self.reward_history.clear()
        self.request_id_counter = 0
        
        # Reset traditional algorithm counters
        self.round_robin_counter = 0
        self.least_connections_history.clear()
        
        return self._get_state()
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Server index to assign the request to (0 to num_servers-1)
            
        Returns:
            observation: Current state
            reward: Reward for this action
            done: Whether episode is finished
            info: Additional information
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
            
        # Generate new request
        new_request = self._generate_request()
        if new_request:
            self.pending_requests.append(new_request)
            
        # Process existing requests
        self._process_requests()
        
        # Assign request to chosen server
        reward = 0.0
        if self.pending_requests:
            request = self.pending_requests.pop(0)
            success = self.servers[action].add_request(
                request['id'], request['size']
            )
            
            if success:
                # Calculate reward based on server state
                reward = self._calculate_reward(action, request)
            else:
                # Penalty for failed assignment
                reward = -10.0
                self.failed_requests += 1
                
        # Update environment
        self.current_step += 1
        self.current_time += 0.1  # Time step
        
        # Check if episode is done
        done = self.current_step >= self.max_requests
        
        # Get observation and info
        observation = self._get_state()
        info = self._get_info()
        
        return observation, reward, done, info
        
    def _generate_request(self) -> Optional[Dict]:
        """Generate a new request based on arrival rate"""
        # Calculate time since last request
        time_since_last = self.current_time - self.last_request_time
        
        # Determine if we should generate a new request
        if self.enable_traffic_spikes:
            # Add traffic spikes
            spike_probability = 0.1
            if np.random.random() < spike_probability:
                arrival_rate = self.request_arrival_rate * 3.0  # 3x spike
            else:
                arrival_rate = self.request_arrival_rate
        else:
            arrival_rate = self.request_arrival_rate
            
        # Exponential distribution for inter-arrival times
        expected_interval = 1.0 / arrival_rate
        if time_since_last >= np.random.exponential(expected_interval):
            self.last_request_time = self.current_time
            
            # Generate request size
            if self.request_size_distribution == "exponential":
                request_size = np.random.exponential(1.0)
            elif self.request_size_distribution == "uniform":
                request_size = np.random.uniform(0.5, 2.0)
            elif self.request_size_distribution == "normal":
                request_size = np.random.normal(1.0, 0.3)
            else:
                request_size = 1.0
                
            request_size = max(0.1, min(5.0, request_size))  # Clamp to reasonable range
            
            self.request_id_counter += 1
            return {
                'id': f"req_{self.request_id_counter}",
                'size': request_size,
                'arrival_time': self.current_time
            }
            
        return None
        
    def _process_requests(self):
        """Process all pending requests across servers"""
        for server in self.servers:
            completed = server.process_requests(self.current_time)
            for request in completed:
                self.completed_requests += 1
                self.response_times.append(request['processing_time'])
                self.completed_requests_history.append(request)
                
    def _calculate_reward(self, server_idx: int, request: Dict) -> float:
        """Calculate reward for assigning request to server"""
        server = self.servers[server_idx]
        metrics = server.get_metrics()
        
        # Base reward for successful assignment
        reward = 1.0
        
        # Penalty for high server utilization
        if metrics.cpu_utilization > 0.8:
            reward -= (metrics.cpu_utilization - 0.8) * 5.0
            
        if metrics.memory_utilization > 0.8:
            reward -= (metrics.memory_utilization - 0.8) * 5.0
            
        # Penalty for long queue
        if metrics.queue_length > 10:
            reward -= (metrics.queue_length - 10) * 0.1
            
        # Bonus for balanced load
        avg_utilization = np.mean([s.get_metrics().cpu_utilization for s in self.servers])
        if abs(metrics.cpu_utilization - avg_utilization) < 0.1:
            reward += 0.5
            
        # Penalty for failed server
        if metrics.status == ServerStatus.FAILED:
            reward -= 20.0
            
        return reward
        
    def _get_state(self) -> np.ndarray:
        """Get current state as numpy array"""
        state = []
        
        # Server states
        for server in self.servers:
            state.extend(server.get_state_vector())
            
        # Global metrics
        total_requests = sum(s.get_metrics().total_requests for s in self.servers)
        failed_requests = sum(s.get_metrics().failed_requests for s in self.servers)
        avg_response_time = np.mean(self.response_times) if self.response_times else 0.0
        
        # Normalize global metrics
        state.extend([
            min(1.0, total_requests / self.max_requests),
            min(1.0, failed_requests / max(1, total_requests)),
            min(1.0, avg_response_time / 1000.0),  # Normalize to seconds
            min(1.0, self.current_step / self.max_requests),
            min(1.0, len(self.pending_requests) / 50.0)  # Normalize queue length
        ])
        
        return np.array(state, dtype=np.float32)
        
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment"""
        metrics = self.get_metrics()
        
        return {
            'total_requests': metrics.total_requests,
            'completed_requests': metrics.completed_requests,
            'failed_requests': metrics.failed_requests,
            'avg_response_time': metrics.avg_response_time,
            'throughput': metrics.throughput,
            'load_balance_fairness': metrics.load_balance_fairness,
            'server_utilizations': metrics.server_utilizations,
            'current_step': self.current_step,
            'max_requests': self.max_requests
        }
        
    def get_metrics(self) -> EnvironmentMetrics:
        """Get current environment metrics"""
        server_metrics = [s.get_metrics() for s in self.servers]
        
        # Calculate throughput
        if self.current_time > 0:
            throughput = self.completed_requests / self.current_time
        else:
            throughput = 0.0
            
        # Calculate load balance fairness (Jain's fairness index)
        utilizations = [m.cpu_utilization for m in server_metrics]
        if utilizations and sum(utilizations) > 0:
            fairness = (sum(utilizations) ** 2) / (len(utilizations) * sum(u ** 2 for u in utilizations))
        else:
            fairness = 1.0
            
        return EnvironmentMetrics(
            total_requests=sum(m.total_requests for m in server_metrics),
            completed_requests=self.completed_requests,
            failed_requests=self.failed_requests,
            avg_response_time=np.mean(self.response_times) if self.response_times else 0.0,
            throughput=throughput,
            load_balance_fairness=fairness,
            server_utilizations=utilizations
        )
        
    def get_traditional_algorithm_metrics(self, algorithm: LoadBalancingAlgorithm) -> Dict[str, float]:
        """Get metrics for traditional load balancing algorithms"""
        if algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return self._simulate_round_robin()
        elif algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return self._simulate_least_connections()
        elif algorithm == LoadBalancingAlgorithm.RANDOM:
            return self._simulate_random()
        else:
            return {}
            
    def _simulate_round_robin(self) -> Dict[str, float]:
        """Simulate round robin algorithm"""
        # This is a simplified simulation
        # In practice, you'd need to replay the entire episode
        return {
            'avg_response_time': np.mean(self.response_times) if self.response_times else 0.0,
            'throughput': self.completed_requests / max(1, self.current_time),
            'fairness': 0.8  # Round robin is generally fair
        }
        
    def _simulate_least_connections(self) -> Dict[str, float]:
        """Simulate least connections algorithm"""
        return {
            'avg_response_time': np.mean(self.response_times) if self.response_times else 0.0,
            'throughput': self.completed_requests / max(1, self.current_time),
            'fairness': 0.9  # Least connections is very fair
        }
        
    def _simulate_random(self) -> Dict[str, float]:
        """Simulate random algorithm"""
        return {
            'avg_response_time': np.mean(self.response_times) if self.response_times else 0.0,
            'throughput': self.completed_requests / max(1, self.current_time),
            'fairness': 0.6  # Random is less fair
        } 