"""
Metrics calculation utilities for RL Load Balancer

This module provides functions for calculating performance metrics
and comparing different load balancing algorithms.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Performance metrics for load balancing"""
    avg_response_time: float
    throughput: float
    fairness: float
    failed_requests: int
    total_requests: int
    server_utilizations: List[float]


def calculate_metrics(
    response_times: List[float],
    throughput: float,
    server_utilizations: List[float],
    failed_requests: int,
    total_requests: int
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics
    
    Args:
        response_times: List of response times
        throughput: Requests per second
        server_utilizations: CPU utilization per server
        failed_requests: Number of failed requests
        total_requests: Total number of requests
        
    Returns:
        PerformanceMetrics object
    """
    avg_response_time = np.mean(response_times) if response_times else 0.0
    
    # Calculate Jain's fairness index for load distribution
    if server_utilizations and sum(server_utilizations) > 0:
        fairness = (sum(server_utilizations) ** 2) / (len(server_utilizations) * sum(u ** 2 for u in server_utilizations))
    else:
        fairness = 1.0
        
    return PerformanceMetrics(
        avg_response_time=avg_response_time,
        throughput=throughput,
        fairness=fairness,
        failed_requests=failed_requests,
        total_requests=total_requests,
        server_utilizations=server_utilizations
    )


def compare_algorithms(
    rl_metrics: PerformanceMetrics,
    round_robin_metrics: PerformanceMetrics,
    least_connections_metrics: PerformanceMetrics,
    random_metrics: PerformanceMetrics
) -> Dict[str, Any]:
    """
    Compare different load balancing algorithms
    
    Args:
        rl_metrics: RL agent performance metrics
        round_robin_metrics: Round robin performance metrics
        least_connections_metrics: Least connections performance metrics
        random_metrics: Random performance metrics
        
    Returns:
        Dictionary with comparison results
    """
    algorithms = {
        'RL': rl_metrics,
        'Round Robin': round_robin_metrics,
        'Least Connections': least_connections_metrics,
        'Random': random_metrics
    }
    
    # Find best performer in each category
    best_response_time = min(algorithms.values(), key=lambda x: x.avg_response_time)
    best_throughput = max(algorithms.values(), key=lambda x: x.throughput)
    best_fairness = max(algorithms.values(), key=lambda x: x.fairness)
    
    comparison = {
        'algorithms': algorithms,
        'best_response_time': {
            'algorithm': [k for k, v in algorithms.items() if v == best_response_time][0],
            'value': best_response_time.avg_response_time
        },
        'best_throughput': {
            'algorithm': [k for k, v in algorithms.items() if v == best_throughput][0],
            'value': best_throughput.throughput
        },
        'best_fairness': {
            'algorithm': [k for k, v in algorithms.items() if v == best_fairness][0],
            'value': best_fairness.fairness
        },
        'improvements': {
            'response_time_improvement': calculate_improvement(
                rl_metrics.avg_response_time,
                min([m.avg_response_time for m in algorithms.values()])
            ),
            'throughput_improvement': calculate_improvement(
                rl_metrics.throughput,
                max([m.throughput for m in algorithms.values()])
            ),
            'fairness_improvement': calculate_improvement(
                rl_metrics.fairness,
                max([m.fairness for m in algorithms.values()])
            )
        }
    }
    
    return comparison


def calculate_improvement(rl_value: float, best_value: float) -> float:
    """
    Calculate improvement percentage of RL over best traditional algorithm
    
    Args:
        rl_value: RL algorithm value
        best_value: Best traditional algorithm value
        
    Returns:
        Improvement percentage (positive = improvement, negative = degradation)
    """
    if best_value == 0:
        return 0.0
        
    return ((rl_value - best_value) / best_value) * 100


def calculate_reward(
    response_time: float,
    server_utilization: float,
    queue_length: int,
    failed: bool,
    target_response_time: float = 100.0,
    max_queue_length: int = 50
) -> float:
    """
    Calculate reward for RL agent based on performance metrics
    
    Args:
        response_time: Current response time
        server_utilization: Current server utilization
        queue_length: Current queue length
        failed: Whether request failed
        target_response_time: Target response time for reward calculation
        max_queue_length: Maximum queue length for normalization
        
    Returns:
        Reward value
    """
    reward = 0.0
    
    # Base reward for successful request
    if not failed:
        reward += 1.0
        
        # Response time reward (lower is better)
        if response_time <= target_response_time:
            reward += 2.0 * (1.0 - response_time / target_response_time)
        else:
            reward -= (response_time - target_response_time) / target_response_time
            
        # Server utilization reward (balanced is better)
        if server_utilization < 0.8:
            reward += 0.5
        else:
            reward -= (server_utilization - 0.8) * 2.0
            
        # Queue length reward (shorter is better)
        normalized_queue = queue_length / max_queue_length
        if normalized_queue < 0.5:
            reward += 0.5
        else:
            reward -= normalized_queue
            
    else:
        # Penalty for failed request
        reward -= 10.0
        
    return reward


def normalize_metrics(metrics: List[float]) -> List[float]:
    """
    Normalize metrics to [0, 1] range
    
    Args:
        metrics: List of metric values
        
    Returns:
        Normalized metrics
    """
    if not metrics:
        return []
        
    min_val = min(metrics)
    max_val = max(metrics)
    
    if max_val == min_val:
        return [0.5] * len(metrics)
        
    return [(val - min_val) / (max_val - min_val) for val in metrics] 