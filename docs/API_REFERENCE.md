# 🔌 JARVIS AI — API Reference & Integration Endpoints

All endpoints are hosted on `http://localhost:8000`. Interactive OpenAPI documentation is accessible at [`http://localhost:8000/docs`](http://localhost:8000/docs).

---

## 1. Core Chat & AI Brain
- **`POST /chat/message`**: Conversational turn with JARVIS.
  ```json
  {
    "message": "Good morning JARVIS",
    "session_id": "optional-uuid"
  }
  ```
- **`GET /chat/history/{session_id}`**: Retrieves stored dialogue turns.
- **`DELETE /chat/session/{session_id}`**: Purges conversation memory for a session.

---

## 2. Autonomous Reasoning Engine
- **`POST /reasoning/solve`**: Executes full 10-step reasoning and execution plan cycle.
  ```json
  {
    "problem": "Configure automated database backup with zero downtime",
    "autonomy_level": 3
  }
  ```
- **`POST /reasoning/analyze`**: Heuristic goal and constraint breakdown.
- **`POST /reasoning/plan`**: Linear subtask dependency generation.

---

## 3. Persistent Memory & Knowledge Base
- **`POST /memory/preferences`**: Save user preference.
  ```json
  {
    "key": "preferred_language",
    "value": "Python",
    "category": "coding",
    "confidence": 1.0
  }
  ```
- **`GET /memory/preferences/{key}`**: Retrieve preference value.
- **`POST /memory/knowledge`**: Store verified fact or research note.
- **`GET /memory/knowledge/search?query=...`**: Semantic vector similarity search.

---

## 4. Internet Research Pipeline
- **`POST /research/topic`**: Deep internet research with cited sources and memory indexing.
  ```json
  {
    "topic": "Quantum Computing advancements in 2026",
    "max_sources": 3
  }
  ```
- **`GET /research/search?q=...`**: Real-time web search query extraction.

---

## 5. Neural Voice Assistant
- **`POST /voice/synthesize`**: Synthesizes and streams raw MP3 neural voice audio.
  ```json
  {
    "text": "All systems are functioning within normal parameters, sir.",
    "voice": "en-GB-RyanNeural"
  }
  ```
- **`POST /voice/process`**: Process conversational turn with wake-word verification.
- **`GET /voice/status`**: Voice engine listener state.

---

## 6. Windows Automation & Hardware Monitoring
- **`GET /automation/system/metrics`**: Real-time CPU, RAM, Disk, and Network telemetry.
- **`POST /automation/app/launch`**: Launches permitted desktop software (`notepad`, `code`, `calc`).
- **`POST /automation/terminal/execute`**: Executes sandboxed shell command with blacklist filter.
  ```json
  {
    "command": "dir",
    "autonomy_level": 4
  }
  ```

---

## 7. Multi-Device Network & Mesh
- **`POST /devices/pairing-code`**: Generates 6-digit one-time device pairing PIN.
- **`POST /devices/pair`**: Authenticates and connects new device (Android / Windows / Linux).
- **`POST /devices/dispatch`**: Sends remote task payload to registered device.
- **`POST /devices/heartbeat`**: Updates device presence and battery state.

---

## 8. Programming Intelligence
- **`POST /programming/analyze`**: Static AST security and syntax analysis.
- **`POST /programming/execute`**: Executes Python script in isolated sandbox.
- **`POST /programming/debug`**: Diagnoses traceback errors and formulates fixes.

---

## 9. Multi-Agent System
- **`GET /agents/list`**: Catalogs all 15 active specialized agents.
- **`POST /agents/orchestrate`**: Coordinates multi-agent collaboration on an objective.
  ```json
  {
    "objective": "Build and secure an API microservice",
    "agents": ["programming", "cybersecurity", "task_planning"]
  }
  ```

---

## 10. Social Media & Vertical Video
- **`POST /social/content/generate`**: Creates 9:16 vertical video script, hook, captions & hashtags.
- **`POST /social/queue/add`**: Queues package for scheduled publishing.
- **`POST /social/publish`**: Dispatches approved posts to target platforms.

---

## 11. Cybersecurity & Kill-Switch
- **`GET /security/audit`**: Inspects tamper-evident SHA-256 hash-chained audit logs.
- **`POST /security/emergency-shutdown`**: Instantly freezes all system automations.
- **`POST /security/disarm?admin_key=jarvis-admin-restore`**: Restores operational status.

---

## 12. Real-Time WebSockets
- **`ws://localhost:8000/ws/stream`**: Bidirectional live HUD telemetry and chat streaming.
- **`ws://localhost:8000/ws/devices`**: Mobile device command socket.
