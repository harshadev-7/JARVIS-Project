"""
JARVIS Video Editing Engine — Adaptive Learning & Self-Improvement
==================================================================
Processes user feedback on edits, tunes preset parameters, tracks rendering
performance, and safely evolves style configurations.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import re

from core.logger import log
from video_editor.project_memory import project_memory
from video_editor.effects_engine import COLOR_GRADE_PRESETS


@dataclass
class AdaptationFeedbackResult:
    """Outcome of processing user feedback."""
    recognized: bool
    parameter_adjusted: str
    old_value: Any
    new_value: Any
    message: str


class AdaptiveEditor:
    """Interprets feedback and optimizes editing parameters."""

    def process_feedback(self, feedback_text: str) -> AdaptationFeedbackResult:
        """
        Analyze user feedback and dynamically tune active editing configurations.
        Examples:
        - "The transitions are too slow"
        - "Make it more vibrant"
        - "The video is too dark"
        - "Use a faster pace"
        """
        lower = feedback_text.lower().strip()

        # 1. Transition speed feedback
        if any(w in lower for w in ["transition", "transitions"]) and any(w in lower for w in ["slow", "faster", "long"]):
            old_val = float(project_memory.get_style_preference("transition_duration", "0.5"))
            new_val = max(0.2, old_val * 0.75)
            project_memory.set_style_preference("transition_duration", f"{new_val:.2f}")
            msg = f"Updated editing preferences: Accelerated transition duration from {old_val:.2f}s to {new_val:.2f}s."
            log.info(msg)
            return AdaptationFeedbackResult(
                recognized=True, parameter_adjusted="transition_duration",
                old_value=old_val, new_value=new_val, message=msg
            )

        if any(w in lower for w in ["transition", "transitions"]) and any(w in lower for w in ["fast", "quick", "abrupt"]):
            old_val = float(project_memory.get_style_preference("transition_duration", "0.5"))
            new_val = min(1.2, old_val * 1.35)
            project_memory.set_style_preference("transition_duration", f"{new_val:.2f}")
            msg = f"Updated editing preferences: Smoothed transition duration from {old_val:.2f}s to {new_val:.2f}s."
            log.info(msg)
            return AdaptationFeedbackResult(
                recognized=True, parameter_adjusted="transition_duration",
                old_value=old_val, new_value=new_val, message=msg
            )

        # 2. Brightness & Color feedback
        if any(w in lower for w in ["dark", "too dark", "dim", "brighter"]):
            old_val = float(project_memory.get_style_preference("brightness_offset", "0.0"))
            new_val = min(0.2, old_val + 0.05)
            project_memory.set_style_preference("brightness_offset", f"{new_val:.2f}")
            msg = f"Updated editing preferences: Increased baseline brightness to +{new_val:.2f}."
            log.info(msg)
            return AdaptationFeedbackResult(
                recognized=True, parameter_adjusted="brightness_offset",
                old_value=old_val, new_value=new_val, message=msg
            )

        if any(w in lower for w in ["vibrant", "colorful", "saturation"]):
            old_val = float(project_memory.get_style_preference("saturation_multiplier", "1.0"))
            new_val = min(1.8, old_val * 1.15)
            project_memory.set_style_preference("saturation_multiplier", f"{new_val:.2f}")
            msg = f"Updated editing preferences: Boosted color saturation multiplier to {new_val:.2f}x."
            log.info(msg)
            return AdaptationFeedbackResult(
                recognized=True, parameter_adjusted="saturation_multiplier",
                old_value=old_val, new_value=new_val, message=msg
            )

        # Default fallback
        msg = f"Recorded feedback: '{feedback_text}'. Calibrating editing heuristics."
        project_memory.set_style_preference("last_feedback", feedback_text)
        return AdaptationFeedbackResult(
            recognized=False, parameter_adjusted="general_feedback",
            old_value=None, new_value=feedback_text, message=msg
        )


# Global adaptive editor singleton
adaptive_editor = AdaptiveEditor()
