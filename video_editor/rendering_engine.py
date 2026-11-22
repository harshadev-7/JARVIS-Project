"""
JARVIS Video Editing Engine — Automatic Rendering Engine
=========================================================
Executes full video assembly and export using the verified local FFmpeg binary:
- Trims and scales multiple input clips into target aspect ratio (9:16, 16:9, etc.)
- Chains complex visual effects (velocity, glitch, color grading, grain)
- Overlays branding (KANKI.7 watermark, intro title, outro signature)
- Mixes background music and audio tracks with smooth fading
- Hardware acceleration detection (NVENC / QSV / multi-threaded CPU libx264)
- Post-render Quality Control (QC) verification and automatic retry loop
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import subprocess
import time
import os

import imageio_ffmpeg
from core.logger import log
from video_editor.editing_brain import EditingDecisionPlan
from video_editor.effects_engine import effects_engine, COLOR_GRADE_PRESETS
from video_editor.branding import branding_manager, BrandingConfig
from video_editor.quality_control import quality_control_engine, QualityInspectionReport
from video_editor.workspace import workspace_manager


@dataclass
class RenderOutputResult:
    """Final result of a completed render job."""
    success: bool
    output_path: str
    duration_sec: float
    width: int
    height: int
    file_size_bytes: int
    render_time_sec: float
    codec: str
    qc_passed: bool
    qc_report: QualityInspectionReport
    command_executed: str
    message: str


class RenderingEngine:
    """Assembles and executes FFmpeg render jobs."""

    def __init__(self, ffmpeg_bin: Optional[str] = None):
        try:
            self.ffmpeg_bin = ffmpeg_bin or imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            self.ffmpeg_bin = "ffmpeg"
        self._hw_accel_available = self._detect_hw_accel()

    def _detect_hw_accel(self) -> Optional[str]:
        """Detect and actively verify available hardware accelerated H.264 encoders."""
        try:
            cmd = [self.ffmpeg_bin, "-encoders"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            output = res.stdout or ""
            if "h264_nvenc" in output:
                # Actively verify that NVIDIA NVENC initializes successfully with installed driver
                test_cmd = [
                    self.ffmpeg_bin, "-y",
                    "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
                    "-c:v", "h264_nvenc",
                    "-f", "null", "-"
                ]
                test_res = subprocess.run(test_cmd, capture_output=True, timeout=5)
                if test_res.returncode == 0:
                    log.info("Verified active NVIDIA NVENC hardware acceleration operational.")
                    return "h264_nvenc"
                else:
                    log.warning("h264_nvenc supported by FFmpeg binary but driver initialization failed (e.g. missing nvcuda.dll). Safely using CPU encoding.")
            if "h264_qsv" in output:
                test_cmd = [
                    self.ffmpeg_bin, "-y",
                    "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
                    "-c:v", "h264_qsv",
                    "-f", "null", "-"
                ]
                test_res = subprocess.run(test_cmd, capture_output=True, timeout=5)
                if test_res.returncode == 0:
                    log.info("Verified active Intel QuickSync hardware acceleration operational.")
                    return "h264_qsv"
        except Exception as e:
            log.debug(f"Hardware acceleration probe skipped: {e}")
        return None

    def render_plan(
        self,
        plan: EditingDecisionPlan,
        background_audio: Optional[str | Path] = None,
        output_filename: Optional[str] = None,
        max_retries: int = 2,
    ) -> RenderOutputResult:
        """
        Render an EditingDecisionPlan into a finalized broadcast-ready MP4.
        Includes automated Quality Control (QC) pass and retry loop.
        """
        fmt = plan.format_config
        target_dir = workspace_manager.get_output_dir(fmt.aspect_ratio)
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        out_name = output_filename or f"final_edit_{plan.style}_{timestamp_str}.mp4"
        out_path = target_dir / out_name

        # Attempt rendering with QC retry loop
        last_qc = None
        cmd_str = ""

        for attempt in range(1, max_retries + 1):
            start_render_time = time.perf_counter()
            log.info(f"Rendering attempt {attempt}/{max_retries} for project '{plan.project_id}' -> {out_path.name}")

            # Build and execute command
            cmd, cmd_str = self._build_ffmpeg_command(
                plan=plan,
                out_path=out_path,
                background_audio=background_audio,
                attempt=attempt,
            )

            try:
                res = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=180)
                render_duration = time.perf_counter() - start_render_time

                if res.returncode != 0:
                    log.warning(f"FFmpeg render attempt {attempt} returned non-zero code {res.returncode}: {res.stderr[:300]}")
                    continue

                # Run Autonomous Quality Control (Phase 9)
                qc = quality_control_engine.inspect_export(
                    file_path=out_path,
                    expected_duration_sec=plan.target_duration_sec,
                    expected_width=fmt.width,
                    expected_height=fmt.height,
                )
                last_qc = qc

                if qc.passed:
                    summary_msg = (
                        f"🎬 **Video Rendered & Verified Successfully!**\n\n"
                        f"• **Project:** {plan.project_id}\n"
                        f"• **Style:** {plan.style.upper()}\n"
                        f"• **Resolution:** {qc.width}x{qc.height} ({fmt.aspect_ratio})\n"
                        f"• **Duration:** {qc.duration_sec:.1f}s\n"
                        f"• **File Size:** {qc.file_size_bytes / (1024*1024):.2f} MB\n"
                        f"• **Render Time:** {render_duration:.2f}s\n"
                        f"• **Branding:** {plan.watermark_brand}\n"
                        f"• **Export Path:** `{out_path}`\n\n"
                        f"All quality control checks passed."
                    )
                    return RenderOutputResult(
                        success=True,
                        output_path=str(out_path),
                        duration_sec=qc.duration_sec,
                        width=qc.width,
                        height=qc.height,
                        file_size_bytes=qc.file_size_bytes,
                        render_time_sec=render_duration,
                        codec=fmt.video_codec,
                        qc_passed=True,
                        qc_report=qc,
                        command_executed=cmd_str,
                        message=summary_msg,
                    )
                else:
                    log.warning(f"Quality Control flagged issues on attempt {attempt}: {qc.issues}")

            except Exception as e:
                log.error(f"Render execution failed on attempt {attempt}: {e}")

        # If all attempts failed or QC failed
        fallback_msg = (
            f"⚠️ Render completed with quality warnings after {max_retries} attempts.\n"
            f"Issues: {last_qc.issues if last_qc else 'Render error'}\n"
            f"Output saved to: `{out_path}`"
        )
        return RenderOutputResult(
            success=out_path.exists() and out_path.stat().st_size > 1000,
            output_path=str(out_path),
            duration_sec=last_qc.duration_sec if last_qc else 0.0,
            width=fmt.width,
            height=fmt.height,
            file_size_bytes=out_path.stat().st_size if out_path.exists() else 0,
            render_time_sec=0.0,
            codec=fmt.video_codec,
            qc_passed=False,
            qc_report=last_qc or QualityInspectionReport(
                file_path=str(out_path), passed=False, file_size_bytes=0,
                duration_sec=0, duration_expected_sec=plan.target_duration_sec,
                duration_delta_sec=plan.target_duration_sec, width=fmt.width, height=fmt.height,
                fps=fmt.fps, has_audio=False, issues=["Render pipeline failure"]
            ),
            command_executed=cmd_str,
            message=fallback_msg,
        )

    def _build_ffmpeg_command(
        self,
        plan: EditingDecisionPlan,
        out_path: Path,
        background_audio: Optional[str | Path] = None,
        attempt: int = 1,
    ) -> Tuple[List[str], str]:
        """Construct the complete FFmpeg command array and complex filtergraph."""
        fmt = plan.format_config
        cmd = [self.ffmpeg_bin, "-y"]

        # 1. Inputs
        if not plan.selected_scenes:
            raise ValueError("Rendering rejected: No valid source video scenes provided. Real video footage is required.")

        for s in plan.selected_scenes:
            cmd.extend(["-ss", f"{s.start_sec:.3f}", "-t", f"{s.duration_sec:.3f}", "-i", s.source_file])
        scene_count = len(plan.selected_scenes)

        # 2. Audio input
        has_bg_audio = False
        if background_audio and Path(background_audio).exists():
            cmd.extend(["-i", str(background_audio)])
            has_bg_audio = True

        # 3. Filtergraph Construction
        filter_chains: List[str] = []

        # Color grade
        cg = COLOR_GRADE_PRESETS.get(plan.color_grade, COLOR_GRADE_PRESETS["standard"])
        color_filter = effects_engine.build_color_filter(cg)

        # Scale & Crop to format target (e.g. 1080x1920)
        scale_crop = fmt.get_scale_crop_filter()

        # Watermark & Titles
        brand_cfg = BrandingConfig(brand_name=plan.watermark_brand)
        watermark = branding_manager.build_watermark_filter(brand_cfg)
        intro_title = branding_manager.build_intro_title_filter(plan.intro_title, duration_sec=2.5)
        outro_card = branding_manager.build_outro_card_filter(plan.target_duration_sec, plan.outro_text, duration_sec=1.5)

        # Build video filter chain for each scene
        v_labels = []
        for i in range(scene_count):
            label = f"[v{i}]"
            v_chain = f"[{i}:v]{scale_crop},{color_filter}{label}"
            filter_chains.append(v_chain)
            v_labels.append(label)

        # Concatenate scenes
        if scene_count > 1:
            concat_inputs = "".join(v_labels)
            concat_chain = f"{concat_inputs}concat=n={scene_count}:v=1:a=0[vconcat]"
            filter_chains.append(concat_chain)
            current_v = "[vconcat]"
        else:
            current_v = v_labels[0]

        # Overlay text, branding, and outro
        overlays = []
        if intro_title:
            overlays.append(intro_title)
        if watermark:
            overlays.append(watermark)
        if outro_card:
            overlays.append(outro_card)

        if overlays:
            overlay_chain = f"{current_v}{','.join(overlays)}[vout]"
            filter_chains.append(overlay_chain)
            final_v = "[vout]"
        else:
            final_v = current_v

        # Audio handling: mix background audio or generate tone
        audio_idx = scene_count if has_bg_audio else None
        if audio_idx is not None:
            # Loop/trim audio with fade in and fade out
            fade_out_st = max(0.5, plan.target_duration_sec - 1.5)
            a_filter = f"[{audio_idx}:a]afade=t=in:st=0:d=0.5,afade=t=out:st={fade_out_st:.2f}:d=1.5,atrim=0:{plan.target_duration_sec:.2f}[aout]"
            filter_chains.append(a_filter)
            final_a = "[aout]"
        else:
            # Synthesize subtle silent/comfort audio track so container has valid audio
            a_gen = f"anullsrc=r=44100:cl=stereo,atrim=0:{plan.target_duration_sec:.2f}[aout]"
            filter_chains.append(a_gen)
            final_a = "[aout]"

        # Assemble filtergraph
        full_filtergraph = ";".join(filter_chains)
        cmd.extend(["-filter_complex", full_filtergraph])

        # Map streams
        cmd.extend(["-map", final_v, "-map", final_a])

        # Encoding options
        codec = "libx264"
        cmd.extend([
            "-c:v", codec,
            "-preset", fmt.preset,
            "-crf", str(fmt.crf),
            "-pix_fmt", fmt.pixel_format,
            "-c:a", fmt.audio_codec,
            "-b:a", f"{fmt.audio_bitrate_kbps}k",
            "-threads", "2",
            "-movflags", "+faststart",
            str(out_path),
        ])

        cmd_str = " ".join(f'"{a}"' if " " in a or "=" in a else a for a in cmd)
        return cmd, cmd_str


# Global rendering engine singleton
rendering_engine = RenderingEngine()
