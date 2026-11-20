"""
JARVIS AI — Real-Time WebSocket Streaming & Device Hub
======================================================
Provides bidirectional live streaming for:
- Holographic Web HUD live telemetry & chat streaming
- Android mobile client remote task and voice command dispatch
- Inter-agent real-time status broadcasts
"""

import asyncio
import json
from typing import List, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.logger import log
from config.settings import settings
from services.automation.system_monitor import system_monitor
from core.brain.jarvis_brain import brain
from core.memory.memory_manager import memory_manager
from core.orchestrator.orchestrator import orchestrator

router = APIRouter(tags=["WebSockets & Live Streaming"])


class ConnectionManager:
    """Manages active WebSocket connections for HUDs and devices."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        log.info(f"WebSocket client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            log.info(f"WebSocket client disconnected. Remaining: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast payload to all connected clients."""
        payload_str = json.dumps(message)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(payload_str)
            except Exception as e:
                log.warning(f"Failed to transmit to client: {e}")
                self.disconnect(connection)


manager = ConnectionManager()


@router.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Primary HUD WebSocket connection:
    - Streams live hardware metrics
    - Handles real-time conversation & command streaming
    """
    await manager.connect(websocket)

    # Send initial welcome packet
    await websocket.send_json({
        "type": "welcome",
        "system": settings.JARVIS_NAME,
        "version": settings.JARVIS_VERSION,
        "status": "operational",
        "autonomy_level": settings.DEFAULT_AUTONOMY_LEVEL,
    })

    try:
        while True:
            data = await websocket.receive_text()
            try:
                packet = json.loads(data)
            except Exception:
                packet = {"type": "chat", "message": data}

            msg_type = packet.get("type", "chat")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong", "status": "alive"})

            elif msg_type == "metrics_request":
                metrics = system_monitor.get_metrics()
                await websocket.send_json({
                    "type": "telemetry",
                    "data": metrics.model_dump(),
                })

            elif msg_type == "chat":
                user_msg = packet.get("message", "")
                session_id = packet.get("session_id", "hud-ws-session")

                memory_manager.add_user_message(session_id, user_msg)
                history = memory_manager.get_history(session_id, max_messages=10)[:-1]

                response = await brain.think(user_msg, conversation_history=history)
                memory_manager.add_assistant_message(session_id, response.content)

                await websocket.send_json({
                    "type": "chat_reply",
                    "reply": response.content,
                    "provider": response.provider,
                    "model": response.model,
                })

            elif msg_type in {"mission", "mission_request"}:
                objective = packet.get("objective", "") or packet.get("message", "")
                if not objective:
                    await websocket.send_json({"type": "mission_error", "message": "Mission objective is required."})
                    continue

                mission = await orchestrator.execute_mission(
                    objective=objective,
                    required_agents=packet.get("agents"),
                    context=packet.get("context") or {},
                )
                await websocket.send_json({
                    "type": "mission_result",
                    "mission": mission.model_dump(mode="json"),
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        log.error(f"WebSocket exception: {e}")
        manager.disconnect(websocket)


@router.websocket("/ws/devices")
async def websocket_devices_endpoint(websocket: WebSocket):
    """
    Android / External Device WebSocket connection:
    - Handles mobile voice commands
    - Receives battery and hardware metrics from devices
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            packet = json.loads(data)

            if packet.get("type") == "voice_command":
                cmd = packet.get("command", "")
                log.info(f"Received mobile voice command: '{cmd}'")
                response = await brain.think(cmd, [])
                await websocket.send_json({
                    "type": "voice_reply",
                    "reply": response.content,
                })

            elif packet.get("type") == "device_hello":
                log.info(f"Device handshake: {packet.get('device_name')} ({packet.get('device_type')})")
                await websocket.send_json({"type": "ack", "message": "Handshake verified."})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        log.error(f"Device WebSocket error: {e}")
        manager.disconnect(websocket)
