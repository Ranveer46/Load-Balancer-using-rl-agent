// RL Load Balancer Dashboard JavaScript

class Dashboard {
    constructor() {
        this.apiBase = 'http://localhost:8000';
        this.updateInterval = null;
        this.rewardChart = null;
        this.responseTimeChart = null;
        this.throughputChart = null;
        this.currentData = {
            metrics: {},
            agent_stats: {},
            episode_history: []
        };
        this.currentTrainingState = false;
        this.loadingOverlayShown = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.initCharts();
        this.startUpdates();
        this.updateConnectionStatus();
        
        // Debug: Check if loading overlay exists
        const loadingOverlay = document.getElementById('loading-overlay');
        console.log('Loading overlay found:', !!loadingOverlay);
        if (loadingOverlay) {
            console.log('Loading overlay classes:', loadingOverlay.className);
        }
    }
    
    setupEventListeners() {
        // Training controls
        document.getElementById('start-training').addEventListener('click', () => {
            this.startTraining();
        });
        
        document.getElementById('stop-training').addEventListener('click', () => {
            this.stopTraining();
        });
        
        // Hide loading button (both in config and overlay)
        const hideLoadingButtons = document.querySelectorAll('#hide-loading');
        hideLoadingButtons.forEach(button => {
            button.addEventListener('click', () => {
                const loadingOverlay = document.getElementById('loading-overlay');
                if (loadingOverlay) {
                    loadingOverlay.classList.remove('show');
                    this.loadingOverlayShown = false;
                    console.log('Loading overlay hidden manually');
                } else {
                    console.error('Loading overlay not found');
                }
            });
        });
        
        // Add test button for debugging
        const testButton = document.createElement('button');
        testButton.textContent = 'Test Loading';
        testButton.className = 'btn btn-secondary';
        testButton.style.marginLeft = '10px';
        testButton.addEventListener('click', () => {
            const loadingOverlay = document.getElementById('loading-overlay');
            if (loadingOverlay) {
                loadingOverlay.classList.toggle('show');
                console.log('Loading overlay toggled:', loadingOverlay.classList.contains('show'));
            } else {
                console.error('Loading overlay not found');
            }
        });
        
        // Add test button to the page
        const buttonGroup = document.querySelector('.button-group');
        if (buttonGroup) {
            buttonGroup.appendChild(testButton);
        }
        
        // Real-time updates
        this.updateInterval = setInterval(() => {
            this.updateMetrics();
        }, 1000);
    }
    
    async updateConnectionStatus() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            if (response.ok) {
                this.setConnectionStatus(true);
            } else {
                this.setConnectionStatus(false);
            }
        } catch (error) {
            this.setConnectionStatus(false);
        }
    }
    
    setConnectionStatus(connected) {
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');
        
        if (connected) {
            statusDot.classList.add('connected');
            statusText.textContent = 'Connected';
        } else {
            statusDot.classList.remove('connected');
            statusText.textContent = 'Disconnected';
        }
    }
    
    async updateMetrics() {
        try {
            const response = await fetch(`${this.apiBase}/metrics`);
            if (response.ok) {
                const data = await response.json();
                this.currentData = data;
                this.updateDashboard(data);
            }
        } catch (error) {
            console.error('Failed to update metrics:', error);
        }
    }
    
    updateDashboard(data) {
        // Update real-time metrics
        document.getElementById('current-episode').textContent = data.current_episode;
        document.getElementById('total-requests').textContent = data.metrics.total_requests || 0;
        document.getElementById('avg-response-time').textContent = 
            `${(data.metrics.avg_response_time || 0).toFixed(1)}ms`;
        document.getElementById('throughput').textContent = 
            `${(data.metrics.throughput || 0).toFixed(1)} req/s`;
        document.getElementById('fairness').textContent = 
            (data.metrics.load_balance_fairness || 0).toFixed(3);
        document.getElementById('failed-requests').textContent = 
            data.metrics.failed_requests || 0;
        
        // Update server status
        this.updateServerStatus();
        
        // Update agent stats
        this.updateAgentStats(data.agent_stats);
        
        // Update charts
        this.updateCharts(data);
        
        // Update episode table
        this.updateEpisodeTable(data.episode_history);
        
        // Update training controls - ensure loading overlay is properly managed
        this.updateTrainingControls(data.is_training);
        
        // Force hide loading overlay if not training
        if (!data.is_training) {
            const loadingOverlay = document.getElementById('loading-overlay');
            if (loadingOverlay) {
                loadingOverlay.classList.remove('show');
                console.log('Loading overlay hidden - training stopped');
            }
        }
    }
    
    async updateServerStatus() {
        try {
            const response = await fetch(`${this.apiBase}/server_status`);
            if (response.ok) {
                const data = await response.json();
                if (data.servers && Array.isArray(data.servers)) {
                    this.renderServerStatus(data.servers);
                } else {
                    console.warn('Invalid server data received:', data);
                }
            }
        } catch (error) {
            console.error('Failed to update server status:', error);
        }
    }
    
    renderServerStatus(servers) {
        const container = document.getElementById('server-status');
        if (!container) {
            console.error('Server status container not found');
            return;
        }
        
        container.innerHTML = '';
        
        if (!servers || !Array.isArray(servers)) {
            container.innerHTML = '<p>No server data available</p>';
            return;
        }
        
        servers.forEach(server => {
            const serverElement = document.createElement('div');
            serverElement.className = `server-item ${server.status}`;
            
            serverElement.innerHTML = `
                <div class="server-id">Server ${server.server_id}</div>
                <div class="server-metrics">
                    <div>CPU: ${(server.cpu_utilization * 100).toFixed(1)}%</div>
                    <div>Memory: ${(server.memory_utilization * 100).toFixed(1)}%</div>
                    <div>Active: ${server.active_requests}</div>
                    <div>Queue: ${server.queue_length}</div>
                    <div>Response: ${server.avg_response_time.toFixed(1)}ms</div>
                </div>
            `;
            
            container.appendChild(serverElement);
        });
    }
    
    updateAgentStats(stats) {
        const container = document.getElementById('agent-stats');
        container.innerHTML = '';
        
        Object.entries(stats).forEach(([key, value]) => {
            const statElement = document.createElement('div');
            statElement.className = 'stat-item';
            
            statElement.innerHTML = `
                <div class="stat-label">${this.formatStatLabel(key)}</div>
                <div class="stat-value">${this.formatStatValue(value)}</div>
            `;
            
            container.appendChild(statElement);
        });
    }
    
    formatStatLabel(key) {
        return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
    formatStatValue(value) {
        if (typeof value === 'number') {
            if (value < 1) return value.toFixed(3);
            if (value < 100) return value.toFixed(1);
            return Math.round(value);
        }
        return value;
    }
    
    initCharts() {
        // Reward chart
        const rewardCtx = document.getElementById('reward-chart').getContext('2d');
        this.rewardChart = new Chart(rewardCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Episode Reward',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(102, 126, 234, 0.1)'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(102, 126, 234, 0.1)'
                        }
                    }
                }
            }
        });
        
        // Response time comparison chart
        this.responseTimeChart = Plotly.newPlot('response-time-chart', [{
            x: ['RL', 'Round Robin', 'Least Connections', 'Random'],
            y: [0, 0, 0, 0],
            type: 'bar',
            marker: {
                color: ['#667eea', '#2ed573', '#ffa502', '#ff4757']
            }
        }], {
            title: 'Average Response Time (ms)',
            height: 250,
            margin: { t: 40, b: 40, l: 60, r: 20 }
        });
        
        // Throughput comparison chart
        this.throughputChart = Plotly.newPlot('throughput-chart', [{
            x: ['RL', 'Round Robin', 'Least Connections', 'Random'],
            y: [0, 0, 0, 0],
            type: 'bar',
            marker: {
                color: ['#667eea', '#2ed573', '#ffa502', '#ff4757']
            }
        }], {
            title: 'Throughput (req/s)',
            height: 250,
            margin: { t: 40, b: 40, l: 60, r: 20 }
        });
    }
    
    updateCharts(data) {
        // Update reward chart
        if (data.episode_history && data.episode_history.length > 0) {
            const episodes = data.episode_history.map(e => e.episode);
            const rewards = data.episode_history.map(e => e.reward);
            
            this.rewardChart.data.labels = episodes;
            this.rewardChart.data.datasets[0].data = rewards;
            this.rewardChart.update();
        }
        
        // Update comparison charts
        this.updateComparisonCharts();
    }
    
    async updateComparisonCharts() {
        try {
            const response = await fetch(`${this.apiBase}/traditional_algorithms`);
            if (response.ok) {
                const algorithms = await response.json();
                
                // Update response time chart
                const responseTimes = [
                    this.currentData.metrics.avg_response_time || 0,
                    algorithms.round_robin?.avg_response_time || 0,
                    algorithms.least_connections?.avg_response_time || 0,
                    algorithms.random?.avg_response_time || 0
                ];
                
                Plotly.update('response-time-chart', {
                    y: [responseTimes]
                });
                
                // Update throughput chart
                const throughputs = [
                    this.currentData.metrics.throughput || 0,
                    algorithms.round_robin?.throughput || 0,
                    algorithms.least_connections?.throughput || 0,
                    algorithms.random?.throughput || 0
                ];
                
                Plotly.update('throughput-chart', {
                    y: [throughputs]
                });
            }
        } catch (error) {
            console.error('Failed to update comparison charts:', error);
        }
    }
    
    updateEpisodeTable(episodes) {
        const tbody = document.getElementById('episode-table-body');
        tbody.innerHTML = '';
        
        // Show last 10 episodes
        const recentEpisodes = episodes.slice(-10).reverse();
        
        recentEpisodes.forEach(episode => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${episode.episode}</td>
                <td>${episode.reward.toFixed(2)}</td>
                <td>${episode.length}</td>
                <td>${episode.avg_response_time.toFixed(1)}ms</td>
                <td>${episode.throughput.toFixed(1)} req/s</td>
                <td>${episode.fairness.toFixed(3)}</td>
            `;
            tbody.appendChild(row);
        });
    }
    
    updateTrainingControls(isTraining) {
        const startBtn = document.getElementById('start-training');
        const stopBtn = document.getElementById('stop-training');
        const loadingOverlay = document.getElementById('loading-overlay');
        
        // Store current training state to prevent unnecessary updates
        if (this.currentTrainingState === isTraining) {
            return; // No change needed
        }
        
        this.currentTrainingState = isTraining;
        
        if (isTraining) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            if (loadingOverlay && !this.loadingOverlayShown) {
                loadingOverlay.classList.add('show');
                this.loadingOverlayShown = true;
                console.log('Loading overlay shown - training started');
            }
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            if (loadingOverlay) {
                loadingOverlay.classList.remove('show');
                this.loadingOverlayShown = false;
                console.log('Loading overlay hidden - training stopped');
            }
        }
    }
    
    async startTraining() {
        const config = {
            agent_type: document.getElementById('agent-type').value,
            num_episodes: parseInt(document.getElementById('num-episodes').value),
            num_servers: parseInt(document.getElementById('num-servers').value),
            request_arrival_rate: parseFloat(document.getElementById('arrival-rate').value),
            learning_rate: 0.001,
            discount_factor: 0.95,
            epsilon: 0.1
        };
        
        try {
            const response = await fetch(`${this.apiBase}/start_training`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            if (response.ok) {
                console.log('Training started successfully');
                // Hide loading overlay after successful start
                setTimeout(() => {
                    const loadingOverlay = document.getElementById('loading-overlay');
                    if (loadingOverlay) {
                        loadingOverlay.classList.remove('show');
                    }
                }, 2000); // Hide after 2 seconds
            } else {
                const error = await response.json();
                console.error('Training start failed:', error);
                alert(`Failed to start training: ${error.detail || 'Unknown error'}`);
                
                // Hide loading overlay on error
                const loadingOverlay = document.getElementById('loading-overlay');
                if (loadingOverlay) {
                    loadingOverlay.classList.remove('show');
                }
            }
        } catch (error) {
            console.error('Failed to start training:', error);
            alert('Failed to start training. Please check the server connection.');
            
            // Hide loading overlay on error
            const loadingOverlay = document.getElementById('loading-overlay');
            if (loadingOverlay) {
                loadingOverlay.classList.remove('show');
            }
        }
    }
    
    async stopTraining() {
        try {
            const response = await fetch(`${this.apiBase}/stop_training`, {
                method: 'POST'
            });
            
            if (response.ok) {
                console.log('Training stopped successfully');
            } else {
                const error = await response.json();
                alert(`Failed to stop training: ${error.detail}`);
            }
        } catch (error) {
            console.error('Failed to stop training:', error);
            alert('Failed to stop training. Please check the server connection.');
        }
    }
    
    startUpdates() {
        // Update connection status every 5 seconds
        setInterval(() => {
            this.updateConnectionStatus();
        }, 5000);
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new Dashboard();
}); 