"""J.A.R.V.I.S. Desktop Arc Reactor HUD Interface Package.

Offline-first desktop graphical user interface with live telemetry,
conversational stream, system dials, and voice control.
"""

from .hud_app import JarvisHUDApp, launch_gui

__all__ = ["JarvisHUDApp", "launch_gui"]
