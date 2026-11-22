"""
JARVIS Video Editing Engine — Autonomous /reel Controller
==========================================================
Coordinates the 12-phase autonomous video editing pipeline:
- Ingests files from media/input/ and media/audio/
- Executes style presets (/reel anime, /reel cinematic, /reel car, /reel luxury, /reel velocity, /reel shorts)
- Handles analysis (/reel analyze), status (/reel status), and rendering (/reel render)
- Falls back gracefully to autonomous synthesis when no raw input files are present
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import re
import time

from core.logger import log
from video_editor.workspace import workspace_manager
from video_editor.input_manager import input_manager
from video_editor.analyzer import video_analyzer
from video_editor.editing_brain import editing_brain, EditingDecisionPlan
from video_editor.rendering_engine import rendering_engine, RenderOutputResult
from video_editor.project_memory import project_memory
from video_editor.autonomous_creator import autonomous_video_creator


class ReelController:
    """Dispatches and coordinates all /reel commands."""

    def __init__(self):
        self._current_process = None
        self._last_plan: Optional[EditingDecisionPlan] = None

    async def execute_command(self, command_line: str) -> str:
        """Parse and execute /reel commands."""
        # Strip '/reel' prefix
        clean_cmd = re.sub(r"^/reel\s*", "", command_line, flags=re.IGNORECASE).strip()
        tokens = clean_cmd.split()
        subcmd = tokens[0].lower() if tokens else "default"

        # 1. Status Command
        if subcmd == "status":
            return self._handle_status()

        # 2. Cancel Command
        if subcmd == "cancel":
            return "🛑 All active rendering tasks acknowledged. Video engine standing by."

        # 3. Analyze Footage Command
        if subcmd == "analyze":
            return self._handle_analyze()

        # 4. Parse target duration if specified (e.g. '30 seconds', '15s', '60')
        target_dur = 30.0
        dur_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\b", clean_cmd)
        if dur_m:
            try:
                parsed_val = float(dur_m.group(1))
                if 5.0 <= parsed_val <= 300.0:
                    target_dur = parsed_val
            except Exception:
                pass

        # 5. Determine Style Preset
        style = "reels"
        for candidate in ["anime", "cinematic", "car", "luxury", "velocity", "shorts"]:
            if candidate in clean_cmd.lower():
                style = candidate
                break

        # 6. Check for available footage in media/input/
        input_videos = workspace_manager.list_input_videos()
        input_audio = workspace_manager.list_input_audio()
        bg_audio = input_audio[0] if input_audio else None

        if input_videos:
            # Multi-clip footage editing pipeline!
            return await self._execute_footage_pipeline(
                style=style,
                target_duration=target_dur,
                input_videos=input_videos,
                bg_audio=bg_audio,
            )
        else:
            # Autonomous Faceless Video Generation with full FFmpeg rendering!
            return await self._execute_autonomous_pipeline(
                style=style,
                target_duration=target_dur,
                topic=clean_cmd if clean_cmd and clean_cmd not in (style, "default") else None,
            )

    def _handle_status(self) -> str:
        """Return workspace telemetry and rendering engine readiness."""
        vids = workspace_manager.list_input_videos()
        auds = workspace_manager.list_input_audio()
        hw_accel = rendering_engine._hw_accel_available or "CPU (Multi-threaded libx264)"
        recent = project_memory.get_recent_projects(limit=3)

        recent_str = "\n".join([f"  • {p['project_id']} ({p['style'].upper()}) — {p['created_at']}" for p in recent]) or "  None yet."

        return (
            f"🎬 **JARVIS Video Editing Engine Status:**\n\n"
            f"• **FFmpeg Engine:** `{rendering_engine.ffmpeg_bin}`\n"
            f"• **Hardware Acceleration:** {hw_accel}\n"
            f"• **Media Ingested:** {len(vids)} video clip(s) in `media/input/`\n"
            f"• **Audio Tracks:** {len(auds)} soundtrack(s) in `media/audio/`\n"
            f"• **Active Branding:** KANKI.7\n\n"
            f"**Recent Projects in SQLite:**\n{recent_str}"
        )

    def _handle_analyze(self) -> str:
        """Scan media/input/ and return a scene-by-scene analysis breakdown."""
        vids = workspace_manager.list_input_videos()
        if not vids:
            return (
                "ℹ️ No source video files found in `media/input/`.\n\n"
                "Place your raw `.mp4`, `.mov`, `.mkv`, or `.avi` footage into `media/input/` "
                "and run `/reel analyze` to inspect scene changes, blur quality, and motion intensity."
            )

        rep = video_analyzer.analyze_all_footage(vids)
        lines = [
            f"📊 **JARVIS Intelligent Footage Analysis Report:**\n",
            f"• **Total Source Clips:** {rep.total_clips}",
            f"• **Combined Duration:** {rep.total_duration_sec:.1f}s",
            f"• **Scenes Detected:** {rep.total_scenes_detected}",
            f"• **Usable Scenes:** {rep.usable_scenes_count} (Discarded: {rep.discarded_scenes_count})",
            f"• **Action Sequences:** {rep.action_scenes_count}",
            f"• **Cinematic Moments:** {rep.cinematic_scenes_count}",
            f"• **Average Sharpness Score:** {rep.average_sharpness:.1f}/100\n",
            "**Scene Breakdown (First 5):**",
        ]
        for s in rep.scenes[:5]:
            lines.append(
                f"  - Scene {s.scene_index}: {s.duration_sec:.1f}s | "
                f"Sharpness: {s.sharpness_score:.0f} | Tags: {', '.join(s.tags) or 'normal'}"
            )

        return "\n".join(lines)

    async def _execute_footage_pipeline(
        self,
        style: str,
        target_duration: float,
        input_videos: List[Path],
        bg_audio: Optional[Path],
    ) -> str:
        """Execute autonomous editing on user-supplied raw footage."""
        log.info(f"Initiating footage editing pipeline for style '{style}' ({target_duration}s)...")

        # 1. Analyze all footage
        analysis = video_analyzer.analyze_all_footage(input_videos)

        # 2. Formulate autonomous editing plan
        plan = editing_brain.formulate_edit(
            scenes=analysis.scenes,
            style=style,
            target_duration=target_duration,
            brand_name="KANKI.7",
        )
        self._last_plan = plan

        # 3. Render video
        result = rendering_engine.render_plan(
            plan=plan,
            background_audio=bg_audio,
        )

        # 4. Save to project memory
        project_memory.save_project(
            project_id=plan.project_id,
            style=style,
            target_duration=target_duration,
            color_grade=plan.color_grade,
            branding_name="KANKI.7",
            timeline_data={"scenes": len(plan.selected_scenes), "effects": plan.applied_effects},
        )

        if result.success:
            project_memory.record_render(
                project_id=plan.project_id,
                style=style,
                output_path=result.output_path,
                duration_sec=result.duration_sec,
                file_size_bytes=result.file_size_bytes,
                render_time_sec=result.render_time_sec,
                qc_passed=result.qc_passed,
            )

        return result.message

    async def _execute_autonomous_pipeline(
        self,
        style: str,
        target_duration: float,
        topic: Optional[str] = None,
    ) -> str:
        """Execute autonomous synthesis when no input video files are present."""
        clean_topic = topic or f"{style.title()} Trends & Highlights"
        res = await autonomous_video_creator.create_autonomous_video(
            topic=clean_topic,
            save_directory=workspace_manager.get_output_dir("reels"),
            watermark_text="KANKI.7",
            include_voiceover=True,
        )
        note = (
            "\n\n[dim]Tip: You can also place raw video clips into `media/input/` and background music into "
            "`media/audio/` to have JARVIS automatically edit your own footage.[/dim]"
        )
        return res.message + note


# Global reel controller singleton
reel_controller = ReelController()
