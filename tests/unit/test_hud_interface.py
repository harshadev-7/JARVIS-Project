"""Unit tests for J.A.R.V.I.S. Desktop Arc Reactor HUD Interface (Phase 19)."""

import pytest
import tkinter as tk
from unittest.mock import MagicMock, patch

from interface.hud_app import JarvisHUDApp


def test_hud_app_headless_initialization():
    """Verify HUD app builds widgets and starts queues without exceptions."""
    try:
        root = tk.Tk()
        root.withdraw()  # Keep hidden during automated test execution
    except Exception:
        pytest.skip("Tkinter display not available in current test environment.")

    try:
        app = JarvisHUDApp(root=root)
        assert app.root is not None
        assert app.COLOR_BG == "#0b0f19"
        assert app.COLOR_ACCENT == "#00f0ff"
        assert app.chat_display is not None
        assert app.input_entry is not None

        # Verify message appending
        app._append_message("JARVIS", "TestAgent", "Unit test message verification.")
        content = app.chat_display.get("1.0", tk.END)
        assert "Unit test message verification" in content

        # Verify task queue processing
        app._task_queue.put(("chat", ("USER", None, "Hello from queue test.")))
        app._process_task_queue()
        updated_content = app.chat_display.get("1.0", tk.END)
        assert "Hello from queue test" in updated_content

    finally:
        app.on_close()
