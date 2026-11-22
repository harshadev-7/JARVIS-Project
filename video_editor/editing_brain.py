"""
JARVIS Video Editing Engine — Autonomous Editing Decision Brain
===============================================================
The cognitive editing core that independently decides:
1. Which clips to use and which to remove
2. The optimal narrative clip arrangement (Hook -> Build -> Climax -> Outro)
3. Transition placement and transition types
4. Where slow motion and speed ramps should occur
5. Synchronization with music beats and audio drops
6. Text overlay placement (Intro hook, dynamic captions, outro signature)
7. Visual effect selection and color grading
8. Conclusion and outro branding
9. Generation of an inspection-ready JSON timeline before rendering
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
import json
import time

from core.logger import log
from video_editor.analyzer import SceneSegment, video_analyzer
from video_editor.branding import BrandingConfig, branding_manager
from video_editor.formats import resolve_format_config, ExportFormatConfig
from video_editor.effects_engine import COLOR_GRADE_PRESETS, ColorGradeConfig
from video_editor.workspace import workspace_manager


@dataclass
class EditingDecisionPlan:
    """The complete autonomous editing blueprint formulated by JARVIS."""
    project_id: str
    style: str
    target_duration_sec: float
    selected_scenes: List[SceneSegment]
    discarded_scenes_count: int
    discarded_reasons: Dict[str, int]
    transitions: List[Dict[str, Any]]
    velocity_ramps: List[Dict[str, Any]]
    beat_sync_events: List[float]
    color_grade: str
    applied_effects: List[str]
    intro_title: str
    watermark_brand: str
    outro_text: str
    timeline_json_path: str
    format_config: ExportFormatConfig


class EditingBrain:
    """Formulates high-level creative and structural editing decisions."""

    def formulate_edit(
        self,
        scenes: List[SceneSegment],
        style: str = "reels",
        target_duration: float = 30.0,
        audio_beats: Optional[List[float]] = None,
        custom_title: Optional[str] = None,
        brand_name: str = "KANKI.7",
    ) -> EditingDecisionPlan:
        """
        Analyze scene options and produce a comprehensive, executable editing decision plan.
        """
        project_id = f"proj-{uuid.uuid4().hex[:8]}"
        style_lower = style.lower()
        fmt_config = resolve_format_config(style_lower)

        # 1. Clip Selection & Filtering
        discarded_reasons = {"blurry": 0, "too_short": 0, "low_energy": 0}
        candidate_scenes: List[SceneSegment] = []

        for s in scenes:
            if s.is_blurry:
                discarded_reasons["blurry"] += 1
            elif s.duration_sec < 0.6:
                discarded_reasons["too_short"] += 1
            else:
                candidate_scenes.append(s)

        # If all were filtered, fallback to all scenes
        if not candidate_scenes:
            candidate_scenes = scenes

        # 2. Select best scenes using the analyzer's heuristic
        selected = video_analyzer.select_best_scenes(
            candidate_scenes, target_duration=target_duration, style=style_lower
        )
        discarded_count = len(scenes) - len(selected)

        # 3. Pacing & Narrative Sequence Arrangement
        # Sort so that highest energy / highest sharpness scene is the Hook (first 3s)
        if len(selected) >= 3:
            # Pick best hook for index 0
            best_hook_idx = max(range(min(4, len(selected))), key=lambda i: selected[i].selection_score)
            if best_hook_idx != 0:
                selected[0], selected[best_hook_idx] = selected[best_hook_idx], selected[0]

        # 4. Transitions
        transitions: List[Dict[str, Any]] = []
        for i in range(len(selected) - 1):
            if "anime" in style_lower or "velocity" in style_lower:
                trans_type = "flash" if (i % 2 == 1) else "wipeleft"
            elif "cinematic" in style_lower or "luxury" in style_lower:
                trans_type = "fade"
            elif "car" in style_lower:
                trans_type = "glitch" if (i % 2 == 1) else "fade"
            else:
                trans_type = "fade"

            transitions.append({
                "from_scene": selected[i].scene_index,
                "to_scene": selected[i+1].scene_index,
                "type": trans_type,
                "duration_sec": 0.3 if "velocity" in style_lower else 0.5,
            })

        # 5. Velocity Ramping & Slow-Mo points
        velocity_ramps: List[Dict[str, Any]] = []
        if "velocity" in style_lower or "car" in style_lower:
            for s in selected:
                if s.is_action_sequence:
                    velocity_ramps.append({
                        "scene_index": s.scene_index,
                        "ramp_type": "speed_up_then_slow_mo",
                        "start_speed": 2.0,
                        "impact_speed": 0.5,
                    })
        elif "cinematic" in style_lower:
            for s in selected:
                if s.is_cinematic and s.duration_sec >= 3.0:
                    velocity_ramps.append({
                        "scene_index": s.scene_index,
                        "ramp_type": "cinematic_slow_mo",
                        "speed": 0.75,
                    })

        # 6. Beat Sync Alignment
        beat_events: List[float] = []
        if audio_beats:
            accumulated_time = 0.0
            for s in selected:
                accumulated_time += s.duration_sec
                # Find closest beat
                closest_beat = min(audio_beats, key=lambda b: abs(b - accumulated_time), default=None)
                if closest_beat and abs(closest_beat - accumulated_time) <= 0.35:
                    beat_events.append(closest_beat)

        # 7. Applied Visual Effects & Color Grading
        color_grade_name = "standard"
        for k in COLOR_GRADE_PRESETS:
            if k in style_lower:
                color_grade_name = k
                break

        applied_fx = [f"Color Grade: {color_grade_name.upper()}"]
        if "cinematic" in style_lower:
            applied_fx.extend(["Cinematic Letterbox (2.39:1)", "Subtle Film Grain (15%)"])
        if "anime" in style_lower:
            applied_fx.extend(["Unsharp Sharpness Mask", "High Saturation Boost (145%)", "Bloom Glow"])
        if "velocity" in style_lower or "car" in style_lower:
            applied_fx.extend(["RGB Glitch Transients", "White Flash Blooms on Drops"])

        # 8. Titles & Branding
        intro_title = custom_title or f"{style.upper()} REEL"
        outro_text = f"EDIT BY {brand_name.upper()}"

        # 9. Save Editable Timeline JSON
        timeline_data = {
            "project_id": project_id,
            "style": style,
            "format": fmt_config.format_name,
            "target_duration_sec": target_duration,
            "color_grade": color_grade_name,
            "branding": {
                "brand_name": brand_name,
                "intro_title": intro_title,
                "outro_text": outro_text,
            },
            "effects": applied_fx,
            "scenes": [s.__dict__ for s in selected],
            "transitions": transitions,
            "velocity_ramps": velocity_ramps,
            "beat_sync_events": beat_events,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        timeline_path = workspace_manager.dirs["editing_timelines"] / f"{project_id}.json"
        timeline_path.write_text(json.dumps(timeline_data, indent=2), encoding="utf-8")
        log.info(f"Formulated autonomous editing plan [{project_id}] -> {timeline_path}")

        return EditingDecisionPlan(
            project_id=project_id,
            style=style,
            target_duration_sec=target_duration,
            selected_scenes=selected,
            discarded_scenes_count=discarded_count,
            discarded_reasons=discarded_reasons,
            transitions=transitions,
            velocity_ramps=velocity_ramps,
            beat_sync_events=beat_events,
            color_grade=color_grade_name,
            applied_effects=applied_fx,
            intro_title=intro_title,
            watermark_brand=brand_name,
            outro_text=outro_text,
            timeline_json_path=str(timeline_path),
            format_config=fmt_config,
        )


# Global editing brain singleton
editing_brain = EditingBrain()
