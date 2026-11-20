# 🚀 JARVIS AI — User & Operator Guide

This manual explains how to run, control, customize, and operate your autonomous JARVIS intelligence system.

---

## 1. Quick Startup Modes

Ensure your Python environment is activated:
```powershell
cd d:\Documents\JARVIS
.venv\Scripts\activate
```

### Mode A: Interactive Holographic Web HUD (Recommended)
Launch the server:
```powershell
.venv\Scripts\python main.py serve
```
Open [`http://localhost:8000/hud`](http://localhost:8000/hud) in your browser:
- **Arc Reactor Visualization**: Live pulsing status indicator.
- **System Telemetry**: Real-time CPU, RAM, Disk telemetry streamed via WebSocket.
- **Voice Activation**: Click **VOICE ON** to enable microphone listening and neural British voice responses (`en-GB-RyanNeural`).
- **Interactive Agent Hub**: Click on any of the 15 agents in the right panel to dispatch operational tasks.
- **Autonomy Controls**: One-click toggles between Autonomy Levels 1 through 5.

---

### Mode B: Terminal Command-Line Interface (CLI)
For direct terminal use:
```powershell
.venv\Scripts\python main.py start
```
Available CLI commands:
- `/solve <task>`: Run the 10-step autonomous reasoning engine.
- `/research <topic>`: Conduct internet research and index findings.
- `/status`: Inspect active provider, autonomy level, and memory sessions.
- `/history`: Display dialogue history.
- `/clear`: Reset conversational session.
- `/exit` or `/quit`: Terminate CLI session.

---

## 2. Configuring Cloud AI Providers

JARVIS includes a built-in **Autonomous Offline Engine** that works immediately without any external API keys.

To connect cloud LLM models, edit [.env](file:///D:/Documents/JARVIS/.env):

### Option 1: Google Gemini (Free / High Speed)
```env
AI_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```

### Option 2: OpenAI (GPT-4o)
```env
AI_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

### Option 3: Local Ollama (100% Free Offline)
```env
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

---

## 3. Autonomy Levels Guide

| Level | Name | Permissions |
|:---:|:---|:---|
| **1** | **Advisor** | Read-only. Answers questions and provides suggestions. Zero system changes. |
| **2** | **Researcher** | Performs web research, inspects files, and accesses memory. |
| **3** | **Planner** | Generates multi-step execution plans and architectures. Requires review before running. |
| **4** | **Executor** | Executes safe system commands, launches apps, and compiles code. |
| **5** | **Autonomous** | Full autonomy across multi-agent workflows, code sandboxing, and device networks. |

---

## 4. Emergency Kill-Switch

If any automation requires an immediate freeze:
- **Engage Kill-Switch**:
  ```bash
  curl -X POST http://localhost:8000/security/emergency-shutdown \
    -H "Content-Type: application/json" \
    -d '{"authorized_by": "operator", "reason": "Immediate security freeze"}'
  ```
- **Disarm & Restore**:
  ```bash
  curl -X POST "http://localhost:8000/security/disarm?admin_key=jarvis-admin-restore"
  ```

---

## 5. Docker Production Deployment

To run JARVIS with full PostgreSQL and Redis clustering:
```bash
cd docker
docker-compose up -d
```
All persistent conversations, memory vectors, and Redis queue tasks will run in dedicated production containers.
