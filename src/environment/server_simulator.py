"""
Server Simulator for RL Load Balancer Environment

This module simulates individual web servers with realistic behavior including:
- CPU and memory utilization
- Request processing with variable latency
- Server health and failure simulation
- Queue management
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ServerStatus(Enum):
    HEALTHY = "healthy"
    OVERLOADED = "overloaded"
    FAILED = "failed"


@dataclass
class ServerMetrics:
    """Server performance metrics"""
    cpu_utilization: float  # 0.0 to 1.0
    memory_utilization: float  # 0.0 to 1.0
    active_requests: int
    queue_length: int
    avg_response_time: float  # milliseconds
    total_requests: int
    failed_requests: int
    status: ServerStatus


class ServerSimulator:
    """
    Simulates a single web server with realistic behavior
    """
    
    def __init__(
        self,
        server_id: int,
        max_cpu: float = 1.0,
        max_memory: float = 1.0,
        base_response_time: float = 50.0,  # ms
        max_queue_size: int = 100,
        failure_probability: float = 0.001
    ):
        self.server_id = server_id
        self.max_cpu = max_cpu
        self.max_memory = max_memory
        self.base_response_time = base_response_time
        self.max_queue_size = max_queue_size
        self.failure_probability = failure_probability
        
        # Current state
        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0
        self.active_requests = 0
        self.queue_length = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.status = ServerStatus.HEALTHY
        self.avg_response_time = self.base_response_time  # Initialize this
        
        # Request processing
        self.request_queue = []
        self.processing_requests = []
        self.request_history = []
        
        # Performance parameters
        self.cpu_per_request = 0.05  # CPU usage per request
        self.memory_per_request = 0.02  # Memory usage per request
        self.cpu_decay_rate = 0.95  # CPU utilization decay
        self.memory_decay_rate = 0.98  # Memory utilization decay
        
        # Timing
        self.last_update = time.time()
        
    def add_request(self, request_id: str, request_size: float = 1.0) -> bool:
        """
        Add a new request to the server
        
        Args:
            request_id: Unique identifier for the request
            request_size: Size/complexity of the request (1.0 = normal)
            
        Returns:
            bool: True if request was accepted, False if rejected
        """
        # Check if server is failed
        if self.status == ServerStatus.FAILED:
            self.failed_requests += 1
            return False
            
        # Check queue capacity
        if self.queue_length >= self.max_queue_size:
            self.failed_requests += 1
            return False
            
        # Simulate server failure
        if np.random.random() < self.failure_probability:
            self.status = ServerStatus.FAILED
            self.failed_requests += 1
            return False
            
        # Add request to queue
        request = {
            'id': request_id,
            'size': request_size,
            'arrival_time': time.time(),
            'processing_time': self._calculate_processing_time(request_size)
        }
        
        self.request_queue.append(request)
        self.queue_length += 1
        self.total_requests += 1
        
        return True
        
    def process_requests(self, current_time: float) -> List[Dict]:
        """
        Process requests and update server state
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of completed requests
        """
        completed_requests = []
        
        # Process active requests
        for request in self.processing_requests[:]:
            if current_time - request['start_time'] >= request['processing_time']:
                # Request completed
                self.processing_requests.remove(request)
                self.active_requests -= 1
                completed_requests.append(request)
                
        # Start processing new requests from queue
        while (self.request_queue and 
               self.active_requests < self._get_max_concurrent_requests()):
            request = self.request_queue.pop(0)
            request['start_time'] = current_time
            self.processing_requests.append(request)
            self.active_requests += 1
            self.queue_length -= 1
            
        # Update server metrics
        self._update_metrics()
        
        return completed_requests
        
    def _calculate_processing_time(self, request_size: float) -> float:
        """Calculate processing time based on request size and server load"""
        base_time = self.base_response_time * request_size
        
        # Add load penalty
        load_penalty = (self.cpu_utilization + self.memory_utilization) * 0.5
        load_penalty *= base_time
        
        # Add some randomness
        noise = np.random.normal(0, base_time * 0.1)
        
        return max(base_time + load_penalty + noise, 10.0)  # Minimum 10ms
        
    def _get_max_concurrent_requests(self) -> int:
        """Get maximum concurrent requests based on server capacity"""
        return max(1, int(10 * (1 - self.cpu_utilization)))
        
    def _update_metrics(self):
        """Update server metrics based on current state"""
        # Update CPU utilization
        cpu_from_requests = self.active_requests * self.cpu_per_request
        cpu_from_queue = self.queue_length * self.cpu_per_request * 0.1
        self.cpu_utilization = min(
            self.max_cpu,
            (self.cpu_utilization * self.cpu_decay_rate + 
             cpu_from_requests + cpu_from_queue)
        )
        
        # Update memory utilization
        memory_from_requests = self.active_requests * self.memory_per_request
        memory_from_queue = self.queue_length * self.memory_per_request * 0.1
        self.memory_utilization = min(
            self.max_memory,
            (self.memory_utilization * self.memory_decay_rate + 
             memory_from_requests + memory_from_queue)
        )
        
        # Update status
        if self.cpu_utilization > 0.9 or self.memory_utilization > 0.9:
            self.status = ServerStatus.OVERLOADED
        elif self.status == ServerStatus.OVERLOADED and self.cpu_utilization < 0.7:
            self.status = ServerStatus.HEALTHY
            
        # Calculate average response time
        if self.request_history:
            response_times = [req['processing_time'] for req in self.request_history[-100:]]
            self.avg_response_time = np.mean(response_times)
        else:
            self.avg_response_time = self.base_response_time
            
    def get_metrics(self) -> ServerMetrics:
        """Get current server metrics"""
        return ServerMetrics(
            cpu_utilization=self.cpu_utilization,
            memory_utilization=self.memory_utilization,
            active_requests=self.active_requests,
            queue_length=self.queue_length,
            avg_response_time=self.avg_response_time,
            total_requests=self.total_requests,
            failed_requests=self.failed_requests,
            status=self.status
        )
        
    def reset(self):
        """Reset server to initial state"""
        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0
        self.active_requests = 0
        self.queue_length = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.status = ServerStatus.HEALTHY
        self.request_queue.clear()
        self.processing_requests.clear()
        self.request_history.clear()
        
    def get_state_vector(self) -> List[float]:
        """Get server state as a vector for RL agent"""
        return [
            self.cpu_utilization,
            self.memory_utilization,
            self.active_requests / 10.0,  # Normalized
            self.queue_length / self.max_queue_size,
            self.avg_response_time / 1000.0,  # Normalized to seconds
            float(self.status == ServerStatus.HEALTHY),
            float(self.status == ServerStatus.OVERLOADED),
            float(self.status == ServerStatus.FAILED)
        ] 