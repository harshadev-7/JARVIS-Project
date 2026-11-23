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
from core.integration.event_bus import SystemEvent, event_bus
from core.integration.subsystem_registry import subsystem_registry
from core.integration.system_diagnostics import system_diagnostics
from core.sentinel import sentinel_daemon

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

# Bridge SystemEventBus to active WebSocket clients
async def _forward_event_to_websockets(event: SystemEvent):
    if manager.active_connections:
        event_dict = event.to_dict()
        await manager.broadcast({
            "type": "system_event",
            "event": event_dict,
        })

event_bus.subscribe("*", _forward_event_to_websockets)


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

            elif msg_type in {"subsystems_request", "subsystems"}:
                all_subsystems = [s.to_dict() for s in subsystem_registry.list_subsystems()]
                probe = subsystem_registry.probe_all_health()
                await websocket.send_json({
                    "type": "subsystems_list",
                    "total": len(all_subsystems),
                    "subsystems": all_subsystems,
                    "health_summary": probe,
                })

            elif msg_type in {"diagnostics_request", "run_diagnostics"}:
                rep = await system_diagnostics.run_full_diagnostics()
                await websocket.send_json({
                    "type": "diagnostics_result",
                    "report": rep.to_dict(),
                    "summary_text": rep.format_terminal_summary(),
                })

            elif msg_type in {"sentinel_request", "sentinel_status"}:
                status = sentinel_daemon.get_status()
                await websocket.send_json({
                    "type": "sentinel_status",
                    "sentinel": status,
                })

            elif msg_type in {"maintenance_request", "run_maintenance"}:
                res = await sentinel_daemon.run_maintenance_now()
                await websocket.send_json({
                    "type": "maintenance_result",
                    "maintenance": res.to_dict(),
                })

            elif msg_type in {"events_history_request", "events_history"}:
                limit = int(packet.get("limit", 20))
                events = [e.to_dict() for e in event_bus.get_history(limit=limit)]
                await websocket.send_json({
                    "type": "events_history",
                    "events": events,
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
