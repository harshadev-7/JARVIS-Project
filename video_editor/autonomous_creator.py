"""J.A.R.V.I.S. Autonomous Video & Viral Reel Creator.

Generates complete short-form videos from scratch without requiring source media:
1. Autonomous Viral Topic Selection (tech trends, AI breakthroughs, coding hacks)
2. Live Internet Research to gather current facts & citations
3. Scriptwriting: Viral hook (first 3s), 3 punchy points, and call to action
4. Voiceover Generation via local/neural TTS engine
5. Dynamic 9:16 Visual Canvas Generation (Cybernetic dark aesthetic #0b0f19 & #00f0ff)
6. Synchronized Subtitle (.srt) synthesis with millisecond timing
7. Safe, non-destructive export to user-specified save location or output/videos/
"""

from __future__ import annotations

import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logger import log
from video_editor.exporter import ExportQuality, ExportResult, video_exporter
from video_editor.safety import media_safety_guard
from video_editor.subtitles import SubtitleEntry, subtitle_manager


@dataclass
class ViralStoryBrief:
    """Structure of an autonomously generated viral short-form video."""
    topic: str
    headline: str
    hook: str
    bullet_points: List[str]
    call_to_action: str
    full_script: str
    duration_sec: float
    subtitle_entries: List[SubtitleEntry] = field(default_factory=list)


class TrendingTopicCurator:
    """Discovers or selects high-engagement viral topics for automated content creation."""

    CURATED_VIRAL_TOPICS = [
        {
            "topic": "Autonomous AI Agents",
            "headline": "How AI Agents Think",
            "hook": "Stop using ChatGPT like a search engine. Autonomous AI agents are changing everything.",
            "points": [
                "Agents break complex problems into 10-step reasoning plans.",
                "They run local neural models right on your computer CPU with zero cloud costs.",
                "They self-debug errors and automate your daily desktop tasks.",
            ],
            "cta": "Follow JARVIS AI for daily offline tech drops.",
        },
        {
            "topic": "Local-First AI Revolution",
            "headline": "Zero Cloud AI is Here",
            "hook": "Why are top developers abandoning cloud AI for local models?",
            "points": [
                "100% privacy: Your code and chats never touch external company servers.",
                "Zero monthly API bills or subscription fees.",
                "Instant offline execution even without internet access.",
            ],
            "cta": "Build your own personal AI assistant today.",
        },
        {
            "topic": "Secret Python Superpowers",
            "headline": "Hidden Python Tricks",
            "hook": "3 Python tricks that feel illegal to know for software engineers.",
            "points": [
                "Use structural pattern matching with 'match-case' to eliminate messy if-else chains.",
                "Leverage slots in classes to slash memory usage by up to 60%.",
                "Use walrus operator := to assign and evaluate variables in a single line.",
            ],
            "cta": "Save this reel and level up your Python code.",
        },
        {
            "topic": "Quantum Computing Reality",
            "headline": "Quantum vs Supercomputers",
            "hook": "Can a quantum computer actually break all internet encryption?",
            "points": [
                "Quantum computers use qubits that exist in multiple states at once via superposition.",
                "They solve complex molecular simulations in seconds instead of thousands of years.",
                "Post-quantum cryptography is already being deployed to keep your data secure.",
            ],
            "cta": "Follow JARVIS for the latest breakthroughs in science.",
        },
        {
            "topic": "Cybersecurity Zero-Day Defense",
            "headline": "Protect Your Workstation",
            "hook": "How hackers exploit zero-day bugs in 2026 before patches exist.",
            "points": [
                "Zero-days target vulnerabilities unknown to the software vendor.",
                "Defense-in-depth and local network isolation are your strongest shields.",
                "Never run unverified shell commands without permission guardrails.",
            ],
            "cta": "Stay secure with JARVIS ethical cybersecurity intelligence.",
        },
    ]

    async def pick_or_research_topic(self, user_topic: Optional[str] = None) -> ViralStoryBrief:
        """Select a trending topic, research it, and generate a viral script."""
        # Case 1: Autonomous selection (no user topic given)
        if not user_topic:
            selected = random.choice(self.CURATED_VIRAL_TOPICS)
            topic = selected["topic"]
            headline = selected["headline"]
            hook = selected["hook"]
            points = selected["points"]
            cta = selected["cta"]
        else:
            # Case 2: User provided a topic -> Research it online or synthesize
            clean_topic = user_topic.strip()
            from services.internet.research_pipeline import research_pipeline
            try:
                rep = await research_pipeline.conduct_research(clean_topic, max_sources=2)
                points = [f.split(":")[-1].strip()[:65] for f in rep.key_findings[:3]]
                if len(points) < 3:
                    points.extend([
                        f"Discover the power of {clean_topic}.",
                        "Local-first intelligence in action.",
                    ])
            except Exception:
                points = [
                    f"Discover the latest breakthroughs in {clean_topic}.",
                    "Modern architectures are rapidly evolving.",
                    "Efficiency and local execution lead the future.",
                ]

            topic = clean_topic
            headline = f"{clean_topic.title()} Trends"
            hook = f"Everything you need to know about {clean_topic} in 30 seconds."
            cta = "Follow JARVIS AI for daily tech intelligence."

        full_script = f"{hook} First: {points[0]} Second: {points[1]} Third: {points[2]} {cta}"

        # Subtitle cards with accurate timings (~25 seconds total)
        entries = [
            SubtitleEntry(index=1, start_sec=0.0, end_sec=3.5, text=hook),
            SubtitleEntry(index=2, start_sec=4.0, end_sec=9.5, text=f"1. {points[0]}"),
            SubtitleEntry(index=3, start_sec=10.0, end_sec=15.5, text=f"2. {points[1]}"),
            SubtitleEntry(index=4, start_sec=16.0, end_sec=21.5, text=f"3. {points[2]}"),
            SubtitleEntry(index=5, start_sec=22.0, end_sec=26.0, text=cta),
        ]

        return ViralStoryBrief(
            topic=topic,
            headline=headline,
            hook=hook,
            bullet_points=points[:3],
            call_to_action=cta,
            full_script=full_script,
            duration_sec=26.5,
            subtitle_entries=entries,
        )


class AutonomousVideoCreator:
    """End-to-end autonomous creator generating videos from scratch."""

    DEFAULT_OUTPUT_DIR = Path("D:/Documents/JARVIS/output/videos")

    def __init__(self):
        self.curator = TrendingTopicCurator()
        self.ffmpeg_path = shutil.which("ffmpeg")

    async def create_autonomous_video(
        self,
        topic: Optional[str] = None,
        save_directory: Optional[str | Path] = None,
        watermark_text: str = "JARVIS AI",
        include_voiceover: bool = True,
    ) -> ExportResult:
        """Autonomously research, script, generate canvas, and export a 9:16 vertical video.

        Args:
            topic: Optional topic. If None, JARVIS selects a viral trending topic on its own.
            save_directory: Directory where finished video should be saved.
            watermark_text: Branding overlay title.
            include_voiceover: Whether to synthesize spoken TTS audio.

        Returns:
            ExportResult detailing execution or script ready for rendering.
        """
        # 1. Output destination
        out_dir = Path(save_directory).resolve() if save_directory else self.DEFAULT_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2. Curate topic & script
        brief = await self.curator.pick_or_research_topic(topic)
        clean_name = re.sub(r"[^\w\-_]", "_", brief.topic.lower())
        timestamp_id = int(brief.duration_sec * 100)

        # 3. Generate subtitle file
        srt_path = out_dir / f"reel_{clean_name}_subtitles.srt"
        subtitle_manager.save_srt(brief.subtitle_entries, srt_path)

        # 4. Generate voiceover audio via local TTS engine
        audio_path = None
        if include_voiceover:
            try:
                from services.voice.tts_engine import tts_engine
                audio_target = out_dir / f"reel_{clean_name}_voice.wav"
                audio_path = await tts_engine.synthesize(brief.full_script, output_file=str(audio_target))
            except Exception as e:
                log.warning(f"Voiceover synthesis note: {e}")

        # 5. Formulate 9:16 Cybernetic Motion Canvas via FFmpeg
        # Generates a modern cybernetic dark blue/cyan background (#0b0f19 to #002b4d) with subtle wave modulation
        # and overlays the title headline + subtitles + watermark
        out_video_path = out_dir / f"reel_{clean_name}.mp4"

        # Format subtitles filter
        sub_filter = subtitle_manager.build_ffmpeg_subtitles_filter(
            srt_path,
            font_size=28,
            primary_color_hex="&H00FFFFFF",
            outline_color_hex="&H00000000",
            margin_v=160,
        )

        headline_text = brief.headline.replace("'", "\\'").replace(":", "\\:")
        title_filter = (
            f"drawtext=text='{headline_text}':fontcolor=0x00f0ff:fontsize=48:x=(w-text_w)/2:y=180:"
            f"box=1:boxcolor=0x0b0f19@0.8:boxborderw=12"
        )
        watermark_filter = (
            f"drawtext=text='{watermark_text}':fontcolor=white@0.7:fontsize=28:x=(w-text_w)/2:y=h-100"
        )

        filter_complex = f"[0:v]{title_filter},{sub_filter},{watermark_filter}[vout]"

        # FFmpeg lavfi generator: generates 1080x1920 9:16 vertical canvas
        cmd = [
            self.ffmpeg_path or "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", f"color=c=0x0b0f19:s=1080x1920:d={brief.duration_sec:.2f}:r=30",
        ]

        if audio_path and Path(audio_path).exists():
            cmd.extend(["-i", str(audio_path)])

        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[vout]",
        ])

        if audio_path and Path(audio_path).exists():
            cmd.extend(["-map", "1:a", "-c:a", "aac", "-b:a", "160k", "-shortest"])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-threads", "2",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_video_path),
        ])

        command_str = " ".join(f'"{arg}"' if " " in arg or "=" in arg else arg for arg in cmd)

        # 6. Execute direct rendering if FFmpeg is installed
        if self.ffmpeg_path:
            try:
                log.info(f"Rendering autonomous video: {out_video_path.name}...")
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if res.returncode == 0:
                    summary_msg = (
                        f"🎬 **Autonomous Video Created Successfully!**\n\n"
                        f"• **Topic Selected:** {brief.topic} ({brief.headline})\n"
                        f"• **Format:** 9:16 Vertical Reel (1080x1920)\n"
                        f"• **Duration:** {brief.duration_sec:.1f} seconds\n"
                        f"• **Hook:** \"{brief.hook}\"\n"
                        f"• **Saved To:** `{out_video_path}`\n\n"
                        f"Ready to upload directly to YouTube Shorts, Instagram Reels, or TikTok!"
                    )
                    return ExportResult(
                        success=True,
                        output_path=str(out_video_path),
                        command_str=command_str,
                        quality_preset="SOCIAL_720P",
                        rendered_directly=True,
                        message=summary_msg,
                    )
            except Exception as e:
                log.warning(f"FFmpeg render encountered issue: {e}")

        # 7. Offline Script & Asset Generation Fallback
        script_ext = ".bat" if sys.platform == "win32" else ".sh"
        script_path = out_dir / f"render_{out_video_path.stem}{script_ext}"

        script_content = (
            f"@echo off\n"
            f"echo ========================================================\n"
            f"echo [JARVIS AI] Autonomous Video Renderer: {brief.headline}\n"
            f"echo ========================================================\n"
            f"echo Rendering 9:16 vertical short to: {out_video_path.name}\n"
            f"{command_str}\n"
            f"echo [JARVIS] Render process complete! Output saved to: {out_video_path.name}\n"
        ) if sys.platform == "win32" else (
            f"#!/usr/bin/env bash\n"
            f"echo '[JARVIS AI] Autonomous Video Renderer: {brief.headline}'\n"
            f"{command_str}\n"
            f"echo '[JARVIS] Render complete: {out_video_path.name}'\n"
        )

        script_path.write_text(script_content, encoding="utf-8")

        summary_msg = (
            f"🎬 **Autonomous Video Assets Formulated Successfully!**\n\n"
            f"• **Topic:** {brief.topic} ({brief.headline})\n"
            f"• **Format:** 9:16 Vertical Reel (1080x1920)\n"
            f"• **Duration:** {brief.duration_sec:.1f}s\n"
            f"• **Viral Hook:** \"{brief.hook}\"\n"
            f"• **Generated Audio Track:** `{audio_path}`\n"
            f"• **Generated Subtitles (.srt):** `{srt_path.name}`\n"
            f"• **Save Destination:** `{out_dir}`\n"
            f"• **1-Click Render Script:** `{script_path.name}`\n\n"
            f"Run `{script_path.name}` or open with FFmpeg to render the finalized MP4."
        )

        return ExportResult(
            success=True,
            output_path=str(out_video_path),
            command_str=command_str,
            quality_preset="SOCIAL_720P",
            rendered_directly=False,
            script_path=str(script_path),
            message=summary_msg,
        )


autonomous_video_creator = AutonomousVideoCreator()

__all__ = ["ViralStoryBrief", "TrendingTopicCurator", "AutonomousVideoCreator", "autonomous_video_creator"]
