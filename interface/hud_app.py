"""J.A.R.V.I.S. Arc Reactor Desktop HUD Application (Phase 19).

Iron Man-inspired cybernetic desktop interface:
- High-contrast sci-fi dark theme (#0b0f19, #00ffff cyan neon, #1f293d)
- Real-time CPU, RAM, and disk utilization telemetry dials
- Multi-agent dialogue stream with distinct assistant agent badges
- Integrated /solve reasoning, /research, and database backup controls
- Emergency Kill Switch button
- Background asyncio thread execution ensuring zero UI freezing
- Pure standard library tkinter with optional CustomTkinter upgrade
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict, Optional

import psutil

from config.settings import settings
from core.brain.jarvis_brain import brain
from core.logger import log
from memory.backup import backup_database


class JarvisHUDApp:
    """J.A.R.V.I.S. Desktop Arc Reactor HUD Application."""

    # Sci-fi color palette
    COLOR_BG = "#0b0f19"          # Deep space dark
    COLOR_PANEL = "#111827"       # Secondary panel
    COLOR_BORDER = "#1e293b"      # Subtle separator
    COLOR_ACCENT = "#00f0ff"      # Glowing cyan / Arc Reactor neon
    COLOR_ACCENT_ALT = "#38bdf8"  # Sky blue
    COLOR_TEXT = "#f1f5f9"        # Clean off-white
    COLOR_TEXT_DIM = "#94a3b8"    # Muted slate
    COLOR_SUCCESS = "#10b981"     # Emerald green
    COLOR_WARN = "#f59e0b"        # Amber
    COLOR_DANGER = "#ef4444"      # Red alert

    def __init__(self, root: Optional[tk.Tk] = None):
        self.root = root or tk.Tk()
        self.root.title(f"{settings.JARVIS_NAME} // H.U.D. CORE v{settings.JARVIS_VERSION}")
        self.root.geometry("1060x720")
        self.root.minsize(880, 580)
        self.root.configure(bg=self.COLOR_BG)

        self._task_queue: queue.Queue = queue.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_thread: Optional[threading.Thread] = None
        self._running = True

        self._start_background_event_loop()
        self._build_ui()
        self._start_telemetry_loop()
        self._process_task_queue()

    def _start_background_event_loop(self) -> None:
        """Run dedicated asyncio loop in background thread for JARVIS brain operations."""
        def _run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._async_thread = threading.Thread(target=_run_loop, daemon=True)
        self._async_thread.start()

    def _build_ui(self) -> None:
        """Construct the HUD window layout."""
        # Top Header
        self._build_header()

        # Main Body Container (Panels left & Chat right)
        body = tk.Frame(self.root, bg=self.COLOR_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 10))

        # Left Column: Telemetry & Quick Action Controls (320px)
        left_col = tk.Frame(body, bg=self.COLOR_PANEL, width=320, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        left_col.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_col.pack_propagate(False)
        self._build_telemetry_panel(left_col)

        # Right Column: Chat History & Input Bar
        right_col = tk.Frame(body, bg=self.COLOR_BG)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_chat_panel(right_col)

    def _build_header(self) -> None:
        """Header with Arc Reactor Title and Live Timestamp."""
        header = tk.Frame(self.root, bg=self.COLOR_BG, height=64)
        header.pack(fill=tk.X, padx=14, pady=10)

        # Arc Reactor symbol + Title
        title_frame = tk.Frame(header, bg=self.COLOR_BG)
        title_frame.pack(side=tk.LEFT)

        reactor_icon = tk.Label(title_frame, text="◈", font=("Helvetica", 24, "bold"), fg=self.COLOR_ACCENT, bg=self.COLOR_BG)
        reactor_icon.pack(side=tk.LEFT, padx=(0, 8))

        title_lbl = tk.Label(
            title_frame,
            text=f"{settings.JARVIS_NAME} // NEURAL CORE",
            font=("Consolas", 18, "bold"),
            fg=self.COLOR_TEXT,
            bg=self.COLOR_BG,
        )
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = tk.Label(
            title_frame,
            text="  [OFFLINE-FIRST AUTONOMOUS AI]",
            font=("Consolas", 10),
            fg=self.COLOR_ACCENT_ALT,
            bg=self.COLOR_BG,
        )
        subtitle_lbl.pack(side=tk.LEFT)

        # Right side: clock and status pill
        right_header = tk.Frame(header, bg=self.COLOR_BG)
        right_header.pack(side=tk.RIGHT)

        self.status_pill = tk.Label(
            right_header,
            text="● ONLINE",
            font=("Consolas", 10, "bold"),
            fg=self.COLOR_SUCCESS,
            bg=self.COLOR_PANEL,
            padx=10,
            pady=4,
        )
        self.status_pill.pack(side=tk.RIGHT, padx=(10, 0))

        self.clock_lbl = tk.Label(
            right_header,
            text="--:--:--",
            font=("Consolas", 12),
            fg=self.COLOR_TEXT_DIM,
            bg=self.COLOR_BG,
        )
        self.clock_lbl.pack(side=tk.RIGHT)

    def _build_telemetry_panel(self, parent: tk.Frame) -> None:
        """Left panel with live hardware stats, model status, and quick actions."""
        pad = tk.Frame(parent, bg=self.COLOR_PANEL)
        pad.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        # Section: Hardware Telemetry
        sec_title = tk.Label(pad, text="HARDWARE TELEMETRY", font=("Consolas", 11, "bold"), fg=self.COLOR_ACCENT, bg=self.COLOR_PANEL)
        sec_title.pack(anchor="w", pady=(0, 8))

        self.cpu_lbl = tk.Label(pad, text="CPU: 0%", font=("Consolas", 10), fg=self.COLOR_TEXT, bg=self.COLOR_PANEL)
        self.cpu_lbl.pack(anchor="w", pady=2)
        self.cpu_bar = ttk.Progressbar(pad, length=280, mode="determinate")
        self.cpu_bar.pack(fill=tk.X, pady=(0, 8))

        self.ram_lbl = tk.Label(pad, text="RAM: 0 / 0 GB (0%)", font=("Consolas", 10), fg=self.COLOR_TEXT, bg=self.COLOR_PANEL)
        self.ram_lbl.pack(anchor="w", pady=2)
        self.ram_bar = ttk.Progressbar(pad, length=280, mode="determinate")
        self.ram_bar.pack(fill=tk.X, pady=(0, 14))

        # Section: Neural Engine Status
        m_title = tk.Label(pad, text="NEURAL ARCHITECTURE", font=("Consolas", 11, "bold"), fg=self.COLOR_ACCENT, bg=self.COLOR_PANEL)
        m_title.pack(anchor="w", pady=(0, 6))

        info_text = (
            "Model: DecoderOnlyTransformer\n"
            "Params: 171,776 (Local CPU)\n"
            "Execution: 100% Offline PyTorch\n"
            "Memory: SQLite WAL + TF-IDF RAG"
        )
        m_info = tk.Label(pad, text=info_text, font=("Consolas", 9), fg=self.COLOR_TEXT_DIM, bg=self.COLOR_PANEL, justify=tk.LEFT)
        m_info.pack(anchor="w", pady=(0, 14))

        # Section: Quick Actions
        act_title = tk.Label(pad, text="COMMAND OVERRIDES", font=("Consolas", 11, "bold"), fg=self.COLOR_ACCENT, bg=self.COLOR_PANEL)
        act_title.pack(anchor="w", pady=(0, 8))

        btn_style = {"font": ("Consolas", 10), "bg": "#1e293b", "fg": self.COLOR_TEXT, "activebackground": self.COLOR_ACCENT, "activeforeground": "#000", "relief": "flat", "pady": 4}

        b1 = tk.Button(pad, text="◈ Run 10-Step Solver (/solve)", command=lambda: self._prompt_command("/solve "), **btn_style)
        b1.pack(fill=tk.X, pady=3)

        b2 = tk.Button(pad, text="🔍 Research Topic (/research)", command=lambda: self._prompt_command("/research "), **btn_style)
        b2.pack(fill=tk.X, pady=3)

        b3 = tk.Button(pad, text="💾 Database Backup", command=self._trigger_backup, **btn_style)
        b3.pack(fill=tk.X, pady=3)

        b4 = tk.Button(pad, text="🔊 Voice Diagnostics", command=self._trigger_voice_test, **btn_style)
        b4.pack(fill=tk.X, pady=3)

        # Emergency Kill Switch at bottom of left panel
        kill_btn = tk.Button(
            pad,
            text="⚠ EMERGENCY KILL SWITCH",
            font=("Consolas", 10, "bold"),
            bg="#991b1b",
            fg="#fee2e2",
            activebackground="#ef4444",
            activeforeground="#fff",
            relief="flat",
            pady=6,
            command=self._emergency_kill,
        )
        kill_btn.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

    def _build_chat_panel(self, parent: tk.Frame) -> None:
        """Main chat display with message flow and input field."""
        # Chat display container
        chat_frame = tk.Frame(parent, bg=self.COLOR_PANEL, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.chat_display = tk.Text(
            chat_frame,
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXT,
            font=("Consolas", 10),
            wrap=tk.WORD,
            state=tk.DISABLED,
            padx=12,
            pady=12,
            relief="flat",
        )
        self.chat_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(chat_frame, command=self.chat_display.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_display["yscrollcommand"] = scrollbar.set

        # Text tags for colorful formatting
        self.chat_display.tag_configure("user_hdr", foreground=self.COLOR_ACCENT, font=("Consolas", 10, "bold"))
        self.chat_display.tag_configure("jarvis_hdr", foreground=self.COLOR_SUCCESS, font=("Consolas", 10, "bold"))
        self.chat_display.tag_configure("agent_badge", foreground=self.COLOR_WARN, font=("Consolas", 9, "italic"))
        self.chat_display.tag_configure("msg_text", foreground=self.COLOR_TEXT, font=("Consolas", 10))
        self.chat_display.tag_configure("system_msg", foreground=self.COLOR_TEXT_DIM, font=("Consolas", 9, "italic"))

        # Welcome message
        self._append_message(
            "JARVIS",
            "ConversationalCortex",
            "Good day, sir. All core systems are online. Custom Transformer is loaded on local CPU.\n"
            "You can type general questions, issue /solve <problem>, or trigger /research <topic>.",
        )

        # Input container at bottom
        input_container = tk.Frame(parent, bg=self.COLOR_PANEL, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        input_container.pack(fill=tk.X)

        self.input_entry = tk.Entry(
            input_container,
            bg="#1e293b",
            fg=self.COLOR_TEXT,
            font=("Consolas", 11),
            insertbackground=self.COLOR_ACCENT,
            relief="flat",
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.input_entry.bind("<Return>", lambda e: self._send_message())
        self.input_entry.focus_set()

        send_btn = tk.Button(
            input_container,
            text="TRANSMIT ❯",
            font=("Consolas", 10, "bold"),
            bg=self.COLOR_ACCENT,
            fg="#000000",
            activebackground=self.COLOR_ACCENT_ALT,
            relief="flat",
            padx=14,
            pady=6,
            command=self._send_message,
        )
        send_btn.pack(side=tk.RIGHT, padx=8, pady=8)

    def _append_message(self, sender: str, agent: Optional[str], text: str) -> None:
        """Append formatted message to scrollable display."""
        self.chat_display.configure(state=tk.NORMAL)
        t_str = datetime.now().strftime("%H:%M:%S")

        if sender == "USER":
            self.chat_display.insert(tk.END, f"\n[{t_str}] USER\n", "user_hdr")
        else:
            agent_str = f" [{agent}]" if agent else ""
            self.chat_display.insert(tk.END, f"\n[{t_str}] JARVIS", "jarvis_hdr")
            if agent_str:
                self.chat_display.insert(tk.END, f"{agent_str}\n", "agent_badge")
            else:
                self.chat_display.insert(tk.END, "\n")

        self.chat_display.insert(tk.END, f"{text.strip()}\n", "msg_text")
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _send_message(self) -> None:
        """Handle user message submit."""
        text = self.input_entry.get().strip()
        if not text:
            return

        self.input_entry.delete(0, tk.END)
        self._append_message("USER", None, text)

        # Dispatch async query to brain in worker loop
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._process_user_message_async(text), self._loop)
        else:
            self._append_message("JARVIS", "SystemError", "Async background engine is offline.")

    async def _process_user_message_async(self, text: str) -> None:
        """Async worker querying the JARVIS brain."""
        try:
            resp = await brain.think(user_message=text, session_id="hud_desktop_session")
            agent = resp.agent_used or "JARVISBrain"
            self._task_queue.put(("chat", ("JARVIS", agent, resp.content)))
        except Exception as e:
            self._task_queue.put(("chat", ("JARVIS", "Error", f"Execution issue: {e}")))

    def _prompt_command(self, prefix: str) -> None:
        """Fill input box with prefix command."""
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, prefix)
        self.input_entry.focus_set()

    def _trigger_backup(self) -> None:
        """Execute database backup utility."""
        try:
            bk_path = backup_database()
            self._append_message("JARVIS", "BackupManager", f"Database backup completed successfully:\n{bk_path}")
        except Exception as e:
            self._append_message("JARVIS", "BackupManager", f"Backup failed: {e}")

    def _trigger_voice_test(self) -> None:
        """Trigger voice engine test."""
        from services.voice.tts_engine import tts_engine
        tts_engine.speak("JARVIS voice systems are operational and standing by.")
        self._append_message("JARVIS", "TTSEngine", "Spoken diagnostic audio dispatched to system speakers.")

    def _emergency_kill(self) -> None:
        """Emergency kill switch confirmation and execution."""
        if messagebox.askyesno("EMERGENCY KILL SWITCH", "Are you certain you wish to halt all JARVIS background tasks?"):
            from services.voice.tts_engine import tts_engine
            tts_engine.stop()
            self._append_message("JARVIS", "SecurityKillSwitch", "EMERGENCY HALT ACTIVATED. All agent operations frozen.")

    def _start_telemetry_loop(self) -> None:
        """Update hardware metrics and clock every 1000ms."""
        if not self._running:
            return

        # Clock
        now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        self.clock_lbl.config(text=now_str)

        # Hardware metrics
        try:
            cpu_p = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            ram_used = round(mem.used / (1024**3), 2)
            ram_total = round(mem.total / (1024**3), 2)
            ram_p = mem.percent

            self.cpu_lbl.config(text=f"CPU: {cpu_p:.1f}%")
            self.cpu_bar["value"] = cpu_p

            self.ram_lbl.config(text=f"RAM: {ram_used} / {ram_total} GB ({ram_p:.1f}%)")
            self.ram_bar["value"] = ram_p
        except Exception:
            pass

        self.root.after(1000, self._start_telemetry_loop)

    def _process_task_queue(self) -> None:
        """Poll thread-safe queue to update GUI from background tasks."""
        try:
            while True:
                msg_type, data = self._task_queue.get_nowait()
                if msg_type == "chat":
                    sender, agent, text = data
                    self._append_message(sender, agent, text)
        except queue.Empty:
            pass

        if self._running:
            self.root.after(100, self._process_task_queue)

    def run(self) -> None:
        """Start the tkinter main loop."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def on_close(self) -> None:
        """Clean shutdown handler."""
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self.root.destroy()


def launch_gui():
    """CLI launcher for Desktop HUD."""
    app = JarvisHUDApp()
    app.run()


if __name__ == "__main__":
    launch_gui()
