# 🏛️ JARVIS AI — Master Architecture Specification

JARVIS (Just A Rather Very Intelligent System) is an autonomous, multi-agent AI platform modeled on the iconic intelligence from Iron Man. It provides continuous reasoning, self-verifying workflows, multi-device networking, and neural voice interaction.

---

## 1. System Architecture Hierarchy

```
                               ┌────────────────────────────────┐
                               │        JARVIS CORE HUB         │
                               └───────────────┬────────────────┘
                                               │
             ┌─────────────────────────────────┼────────────────────────────────┐
             ▼                                 ▼                                ▼
   [Multi-Agent Mesh]               [Reasoning Engine]              [Persistent Memory]
    • 15 Domain Agents               • 10-Step Solver                • Episodic Memory
    • Inter-Agent Messaging          • Heuristic Decomposition       • Vector Semantic Store
    • Mission Synthesizer            • RCA & Outcome Verifier        • Mistake Avoidance DB
             │                                 │                                │
             └─────────────────────────────────┼────────────────────────────────┘
                                               │
             ┌─────────────────────────────────┴────────────────────────────────┐
             ▼                                                                  ▼
    [Services & Devices]                                              [User Interfaces]
     • Windows Automation Engine                                       • Holographic Arc Reactor HUD
     • Android Remote Node & Mesh                                      • Interactive Rich CLI
     • HD Neural Voice (Edge-TTS)                                      • REST & WebSocket Streaming
     • Cryptographic Audit Logger                                      • Emergency Kill-Switch
```

---

## 2. The 15 Specialized AI Agents

Every agent inherits from `BaseAgent` and implements dedicated domain capabilities:

| # | Agent | Primary Role & Capabilities |
|---|---|---|
| 1 | **Research Agent** | Real-time web search, HTML cleaning, factual synthesis, and source verification. |
| 2 | **Problem-Solving Agent** | 10-step reasoning, goal decomposition, strategy evaluation, and root-cause analysis. |
| 3 | **Programming Agent** | AST static code security linting, function extraction, and sandboxed execution. |
| 4 | **Mathematical Agent** | Formula evaluation, geometric and trigonometric calculations, and numeric projections. |
| 5 | **Data Analysis Agent** | Statistical modeling, dataset summary, anomaly detection, and correlation analysis. |
| 6 | **Automation Agent** | Windows OS telemetry, application launching, and sandboxed terminal command execution. |
| 7 | **Device Management Agent** | Cross-device mesh pairing (6-digit PIN handshake), heartbeat telemetry, and remote task dispatch. |
| 8 | **Social Media Agent** | Multi-platform publishing scheduler, approval workflows, and posting queue dispatch. |
| 9 | **Content Creation Agent** | Viral hooks, 15s/45s script generation, high-CTR thumbnail prompts, and platform captions. |
| 10 | **Video Editing Agent** | 9:16 vertical video specifications (YouTube Shorts / IG Reels / TikTok) and FFmpeg pipelines. |
| 11 | **Personal Assistant** | Schedule management, action item tracking, task reminders, and daily agenda briefings. |
| 12 | **Memory Management Agent** | Episodic conversation indexing, user preference storage, and semantic vector retrieval. |
| 13 | **Cybersecurity Agent** | Cryptographic hash-chained audit logging, command blacklist enforcement, and instant emergency kill-switch. |
| 14 | **Communication Agent** | High-definition neural speech synthesis (Edge-TTS `en-GB-RyanNeural`) and wake-word detection. |
| 15 | **Task Planning Agent** | Dependency graph generation, linear subtask breakdown, and milestone verification. |

---

## 3. The 10-Step Autonomous Reasoning Engine

Complex missions are resolved through a deterministic 10-step problem-solving sequence:
1. **Objective Understanding**: Extracts core goal, implicit constraints, and detected uncertainties.
2. **Situation Assessment**: Surveys current hardware, memory, and environmental state.
3. **Goal Decomposition**: Breaks the objective into ordered subtasks with clear dependency chains.
4. **Approach Generation**: Generates 3 distinct candidate strategies (e.g. conservative, balanced, aggressive).
5. **Autonomy Permission Gating**: Checks the action against the current Autonomy Level (1 to 5).
6. **Approach Evaluation & Scoring**: Computes composite scores across feasibility, risk, and speed.
7. **Strategy Selection**: Picks the highest-scoring candidate path.
8. **Execution Plan Construction**: Generates executable task parameters and milestones.
9. **Outcome Verification**: Validates observed outcome against expected success criteria.
10. **Root Cause Analysis & Fallback**: Formulates immediate self-healing remedies if any step fails.

---

## 4. Multi-Tier Memory Taxonomy

1. **Short-Term Session Memory**: In-memory message ring buffer maintained per session ID.
2. **Long-Term Episodic Memory**: SQLite/PostgreSQL `conversation_episodes` table tracking interaction history.
3. **User Preferences**: Confidence-weighted key-value store for user habits, tools, and styles.
4. **Mistake Avoidance Database**: Tracks failed commands and root causes to prevent repeating errors.
5. **Semantic Vector Store**: Cosine similarity vector search over verified research and knowledge notes.

---

## 5. Security & Isolation Architecture

- **Tamper-Evident Audit Trail**: Every high-privilege action is SHA-256 hash-chained with the preceding event hash (`audit_logger`).
- **Emergency Kill-Switch**: `POST /security/emergency-shutdown` immediately halts all automation and command execution until authenticated restore via admin key.
- **Role-Based Access Control**: Strict hierarchy (Admin > Operator > Auditor > Readonly).
- **Token Bucket Rate Limiting**: Prevents API flooding and brute-force attempts.
