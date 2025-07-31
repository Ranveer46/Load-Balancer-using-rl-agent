"""
WebSocket support for real-time dashboard updates

This module provides WebSocket functionality for pushing real-time
training metrics to the dashboard frontend.
"""

import asyncio
import json
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket"""
        if websocket.state == WebSocketState.CONNECTED:
            await websocket.send_text(message)
            
    async def broadcast(self, message: str):
        """Broadcast message to all connected WebSockets"""
        disconnected = []
        for connection in self.active_connections:
            try:
                if connection.state == WebSocketState.CONNECTED:
                    await connection.send_text(message)
                else:
                    disconnected.append(connection)
            except:
                disconnected.append(connection)
                
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for heartbeat
            await manager.send_personal_message(f"Message: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def broadcast_metrics(metrics: Dict[str, Any]):
    """Broadcast metrics to all connected clients"""
    message = json.dumps({
        "type": "metrics_update",
        "data": metrics,
        "timestamp": asyncio.get_event_loop().time()
    })
    await manager.broadcast(message)


async def broadcast_training_status(status: Dict[str, Any]):
    """Broadcast training status to all connected clients"""
    message = json.dumps({
        "type": "training_status",
        "data": status,
        "timestamp": asyncio.get_event_loop().time()
    })
    await manager.broadcast(message)


async def broadcast_server_status(servers: Dict[str, Any]):
    """Broadcast server status to all connected clients"""
    message = json.dumps({
        "type": "server_status",
        "data": servers,
        "timestamp": asyncio.get_event_loop().time()
    })
    await manager.broadcast(message) 