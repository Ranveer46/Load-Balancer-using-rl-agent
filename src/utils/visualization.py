"""
Visualization utilities for RL Load Balancer

This module provides functions for creating plots and visualizations
of training progress and algorithm comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_training_progress(
    episode_rewards: List[float],
    episode_lengths: List[int],
    avg_response_times: List[float],
    throughputs: List[float],
    save_path: str = None
) -> go.Figure:
    """
    Create a comprehensive training progress plot
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        avg_response_times: List of average response times
        throughputs: List of throughput values
        save_path: Path to save the plot
        
    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Episode Lengths', 'Response Times', 'Throughput'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    episodes = list(range(1, len(episode_rewards) + 1))
    
    # Episode rewards
    fig.add_trace(
        go.Scatter(x=episodes, y=episode_rewards, mode='lines', name='Reward',
                  line=dict(color='#667eea', width=2)),
        row=1, col=1
    )
    
    # Episode lengths
    fig.add_trace(
        go.Scatter(x=episodes, y=episode_lengths, mode='lines', name='Length',
                  line=dict(color='#2ed573', width=2)),
        row=1, col=2
    )
    
    # Response times
    fig.add_trace(
        go.Scatter(x=episodes, y=avg_response_times, mode='lines', name='Response Time',
                  line=dict(color='#ffa502', width=2)),
        row=2, col=1
    )
    
    # Throughput
    fig.add_trace(
        go.Scatter(x=episodes, y=throughputs, mode='lines', name='Throughput',
                  line=dict(color='#ff4757', width=2)),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Training Progress',
        height=600,
        showlegend=False,
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
        
    return fig


def plot_comparison(
    algorithms: Dict[str, Any],
    save_path: str = None
) -> go.Figure:
    """
    Create comparison plots for different algorithms
    
    Args:
        algorithms: Dictionary with algorithm metrics
        save_path: Path to save the plot
        
    Returns:
        Plotly figure object
    """
    algo_names = list(algorithms.keys())
    response_times = [algorithms[name].avg_response_time for name in algo_names]
    throughputs = [algorithms[name].throughput for name in algo_names]
    fairness_scores = [algorithms[name].fairness for name in algo_names]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Response Time (ms)', 'Throughput (req/s)', 'Fairness'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#667eea', '#2ed573', '#ffa502', '#ff4757']
    
    # Response times
    fig.add_trace(
        go.Bar(x=algo_names, y=response_times, name='Response Time',
               marker_color=colors[:len(algo_names)]),
        row=1, col=1
    )
    
    # Throughput
    fig.add_trace(
        go.Bar(x=algo_names, y=throughputs, name='Throughput',
               marker_color=colors[:len(algo_names)]),
        row=1, col=2
    )
    
    # Fairness
    fig.add_trace(
        go.Bar(x=algo_names, y=fairness_scores, name='Fairness',
               marker_color=colors[:len(algo_names)]),
        row=1, col=3
    )
    
    fig.update_layout(
        title='Algorithm Comparison',
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
        
    return fig


def plot_server_utilization(
    server_utilizations: List[List[float]],
    server_names: List[str] = None,
    save_path: str = None
) -> go.Figure:
    """
    Create server utilization heatmap
    
    Args:
        server_utilizations: List of utilization values per server
        server_names: Names of servers
        save_path: Path to save the plot
        
    Returns:
        Plotly figure object
    """
    if server_names is None:
        server_names = [f'Server {i}' for i in range(len(server_utilizations[0]))]
        
    fig = go.Figure(data=go.Heatmap(
        z=server_utilizations,
        x=server_names,
        y=list(range(len(server_utilizations))),
        colorscale='RdYlGn_r',
        zmin=0,
        zmax=1
    ))
    
    fig.update_layout(
        title='Server Utilization Over Time',
        xaxis_title='Servers',
        yaxis_title='Time Steps',
        height=400,
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
        
    return fig


def plot_reward_distribution(
    rewards: List[float],
    save_path: str = None
) -> go.Figure:
    """
    Create reward distribution histogram
    
    Args:
        rewards: List of reward values
        save_path: Path to save the plot
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=go.Histogram(
        x=rewards,
        nbinsx=30,
        marker_color='#667eea',
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Reward Distribution',
        xaxis_title='Reward',
        yaxis_title='Frequency',
        height=400,
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
        
    return fig


def create_dashboard_plots(
    episode_data: List[Dict[str, Any]],
    server_data: List[Dict[str, Any]],
    agent_stats: Dict[str, Any]
) -> Dict[str, go.Figure]:
    """
    Create all dashboard plots
    
    Args:
        episode_data: List of episode data dictionaries
        server_data: List of server data dictionaries
        agent_stats: Agent statistics dictionary
        
    Returns:
        Dictionary of plotly figures
    """
    plots = {}
    
    # Extract data
    episodes = [e['episode'] for e in episode_data]
    rewards = [e['reward'] for e in episode_data]
    response_times = [e['avg_response_time'] for e in episode_data]
    throughputs = [e['throughput'] for e in episode_data]
    
    # Reward progress
    plots['reward_progress'] = go.Figure(data=go.Scatter(
        x=episodes, y=rewards, mode='lines',
        line=dict(color='#667eea', width=2)
    ))
    plots['reward_progress'].update_layout(
        title='Training Progress - Rewards',
        xaxis_title='Episode',
        yaxis_title='Reward',
        height=300,
        template='plotly_white'
    )
    
    # Response time progress
    plots['response_time_progress'] = go.Figure(data=go.Scatter(
        x=episodes, y=response_times, mode='lines',
        line=dict(color='#ffa502', width=2)
    ))
    plots['response_time_progress'].update_layout(
        title='Training Progress - Response Time',
        xaxis_title='Episode',
        yaxis_title='Response Time (ms)',
        height=300,
        template='plotly_white'
    )
    
    # Throughput progress
    plots['throughput_progress'] = go.Figure(data=go.Scatter(
        x=episodes, y=throughputs, mode='lines',
        line=dict(color='#2ed573', width=2)
    ))
    plots['throughput_progress'].update_layout(
        title='Training Progress - Throughput',
        xaxis_title='Episode',
        yaxis_title='Throughput (req/s)',
        height=300,
        template='plotly_white'
    )
    
    # Server utilization
    if server_data:
        server_ids = [s['server_id'] for s in server_data]
        utilizations = [s['cpu_utilization'] for s in server_data]
        
        plots['server_utilization'] = go.Figure(data=go.Bar(
            x=server_ids, y=utilizations,
            marker_color='#667eea'
        ))
        plots['server_utilization'].update_layout(
            title='Current Server Utilization',
            xaxis_title='Server ID',
            yaxis_title='CPU Utilization',
            height=300,
            template='plotly_white'
        )
    
    return plots


def save_matplotlib_plots(
    episode_rewards: List[float],
    episode_lengths: List[int],
    save_path: str = None
):
    """
    Create matplotlib plots for offline analysis
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        save_path: Path to save the plots
    """
    plt.style.use('seaborn-v0_8')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Reward plot
    ax1.plot(episode_rewards, color='#667eea', linewidth=2)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    # Length plot
    ax2.plot(episode_lengths, color='#2ed573', linewidth=2)
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Length')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 