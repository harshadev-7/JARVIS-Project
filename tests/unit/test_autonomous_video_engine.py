"""
JARVIS Autonomous Video Editing Engine — Comprehensive Test Suite
==================================================================
Tests all 12 phases of the autonomous video editing engine:
- Phase 1: Workspace directories & video input metadata manager
- Phase 2: Video & scene analyzer, blur detection, scene selection
- Phase 3: Autonomous editing decision brain & JSON timeline generation
- Phase 4: Audio waveform analysis & beat synchronization
- Phase 5: Professional video effects engine (color grading, velocity, glitch)
- Phase 6: KANKI.7 branding, watermarks, intro/outro cards
- Phase 7: Multi-format presets (Reels 9:16, Shorts 9:16, Cinematic 16:9, Square 1:1)
- Phase 8: Real FFmpeg rendering engine
- Phase 9: Quality Control (QC) inspection pass
- Phase 10: /reel command suite dispatcher
- Phase 11: SQLite project memory & render history
- Phase 12: Adaptive feedback learning & preset calibration
"""

import pytest
import subprocess
from pathlib import Path

from video_editor.workspace import WorkspaceManager
from video_editor.input_manager import VideoInputManager, VideoMediaMetadata
from video_editor.analyzer import VideoAnalyzer, SceneSegment, FootageAnalysisReport
from video_editor.editing_brain import EditingBrain, EditingDecisionPlan
from video_editor.effects_engine import EffectsEngine, COLOR_GRADE_PRESETS
from video_editor.branding import BrandingManager, BrandingConfig
from video_editor.formats import resolve_format_config, FORMAT_PRESETS
from video_editor.quality_control import QualityControlEngine, QualityInspectionReport
from video_editor.rendering_engine import RenderingEngine
from video_editor.project_memory import ProjectMemory
from video_editor.adaptive_editor import AdaptiveEditor
from video_editor.reel_controller import ReelController


@pytest.fixture
def test_workspace(tmp_path):
    """Create isolated test workspace directories."""
    ws = WorkspaceManager(root=tmp_path)
    return ws


@pytest.fixture
def synthetic_video(tmp_path):
    """Generate a clean synthetic MP4 clip using local FFmpeg for reliable testing."""
    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    out_video = tmp_path / "test_sample_clip.mp4"

    # 3.0 second 1080x1920 test video with audio tone
    cmd = [
        ffmpeg_bin, "-y",
        "-f", "lavfi", "-i", "testsrc=size=1080x1920:rate=30:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        str(out_video),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out_video


# ── Phase 1: Workspace & Input Manager Tests ─────────────────────────────────

def test_workspace_directory_structure(test_workspace):
    """Verify all workspace directories are created."""
    for key, dpath in test_workspace.dirs.items():
        assert dpath.exists()
        assert dpath.is_dir()


def test_input_manager_probing_and_thumbnails(synthetic_video, tmp_path):
    """Verify metadata probing and thumbnail extraction."""
    mgr = VideoInputManager()
    meta = mgr.probe_video(synthetic_video)

    assert meta.filename == "test_sample_clip.mp4"
    assert meta.width == 1080
    assert meta.height == 1920
    assert meta.fps == 30.0
    assert abs(meta.duration_sec - 3.0) < 0.5
    assert meta.has_audio is True
    assert meta.aspect_ratio == "9:16"

    # Test thumbnail generation
    thumb = mgr.generate_thumbnail(synthetic_video, timestamp_sec=1.0, width=320)
    assert thumb.exists()
    assert thumb.stat().st_size > 500


# ── Phase 2: Video Analysis Tests ────────────────────────────────────────────

def test_video_analyzer_scene_detection(synthetic_video):
    """Verify scene detection and segment scoring."""
    analyzer = VideoAnalyzer()
    scenes = analyzer.analyze_clip(synthetic_video, min_scene_len=0.5)

    assert len(scenes) >= 1
    s0 = scenes[0]
    assert isinstance(s0, SceneSegment)
    assert s0.duration_sec > 0.0
    assert s0.selection_score > 0.0

    # Test scene selection algorithm
    selected = analyzer.select_best_scenes(scenes, target_duration=2.0, style="anime")
    assert len(selected) >= 1
    assert selected[0].duration_sec <= 2.5


# ── Phase 3: Autonomous Editing Brain Tests ──────────────────────────────────

def test_editing_brain_decision_plan(synthetic_video):
    """Verify autonomous decision engine builds an executable EditingDecisionPlan."""
    analyzer = VideoAnalyzer()
    scenes = analyzer.analyze_clip(synthetic_video)

    brain = EditingBrain()
    plan = brain.formulate_edit(
        scenes=scenes,
        style="anime",
        target_duration=3.0,
        brand_name="KANKI.7",
    )

    assert isinstance(plan, EditingDecisionPlan)
    assert plan.style == "anime"
    assert plan.color_grade == "anime"
    assert plan.watermark_brand == "KANKI.7"
    assert len(plan.selected_scenes) >= 1
    assert Path(plan.timeline_json_path).exists()


# ── Phase 4, 5, 6, 7: Effects, Branding & Formats Tests ─────────────────────

def test_effects_engine_filtergraphs():
    """Verify programmatic FFmpeg filter expressions."""
    fx = EffectsEngine()
    cg = COLOR_GRADE_PRESETS["anime"]
    color_filt = fx.build_color_filter(cg)
    assert "eq=" in color_filt
    assert "colorbalance=" in color_filt

    v_vel, a_vel = fx.build_velocity_filter(speed_factor=2.0)
    assert "setpts=0.5000*PTS" in v_vel
    assert "atempo=2.000" in a_vel

    letterbox = fx.build_letterbox_filter("cinematic")
    assert "drawbox=" in letterbox


def test_branding_manager_kanki_default():
    """Verify default KANKI.7 branding overlays."""
    bm = BrandingManager()
    assert bm.config.brand_name == "KANKI.7"

    wm_filter = bm.build_watermark_filter()
    assert "KANKI.7" in wm_filter
    assert "drawtext=" in wm_filter

    intro_filter = bm.build_intro_title_filter("VIRAL ANIME")
    assert "VIRAL ANIME" in intro_filter

    outro_filter = bm.build_outro_card_filter(total_duration_sec=30.0)
    assert "EDIT BY KANKI.7" in outro_filter


def test_multi_format_presets():
    """Verify format specifications."""
    reels = resolve_format_config("reels")
    assert reels.width == 1080 and reels.height == 1920
    assert reels.aspect_ratio == "9:16"

    cinematic = resolve_format_config("cinematic")
    assert cinematic.width == 1920 and cinematic.height == 1080
    assert cinematic.aspect_ratio == "16:9"

    square = resolve_format_config("square")
    assert square.width == 1080 and square.height == 1080
    assert square.aspect_ratio == "1:1"


# ── Phase 8 & 9: Rendering Engine & Quality Control Tests ────────────────────

def test_rendering_engine_and_quality_control(synthetic_video, tmp_path):
    """Verify real FFmpeg rendering with bundled binary and Quality Control pass."""
    analyzer = VideoAnalyzer()
    scenes = analyzer.analyze_clip(synthetic_video)

    brain = EditingBrain()
    plan = brain.formulate_edit(
        scenes=scenes,
        style="reels",
        target_duration=3.0,
        brand_name="KANKI.7",
    )

    renderer = RenderingEngine()
    out_file = tmp_path / "test_render_output.mp4"

    res = renderer.render_plan(
        plan=plan,
        output_filename=str(out_file),
        max_retries=1,
    )

    assert res.success is True
    assert Path(res.output_path).exists()
    assert res.file_size_bytes > 5000

    # Verify Quality Control pass
    qc_engine = QualityControlEngine()
    qc_report = qc_engine.inspect_export(
        file_path=res.output_path,
        expected_duration_sec=3.0,
        expected_width=1080,
        expected_height=1920,
    )
    assert qc_report.passed is True
    assert qc_report.has_audio is True
    assert qc_report.black_frames_detected == 0


# ── Phase 10: /reel Controller Tests ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_reel_controller_status_and_analyze():
    """Verify /reel status and /reel analyze commands."""
    ctrl = ReelController()
    status_msg = await ctrl.execute_command("/reel status")
    assert "JARVIS Video Editing Engine Status" in status_msg
    assert "KANKI.7" in status_msg

    analyze_msg = await ctrl.execute_command("/reel analyze")
    assert "Footage Analysis Report" in analyze_msg or "media/input/" in analyze_msg


# ── Phase 11 & 12: Project Memory & Adaptive Learning Tests ──────────────────

def test_project_memory_and_adaptive_editor(tmp_path):
    """Verify SQLite persistence and adaptive feedback tuning."""
    mem_db = tmp_path / "test_memory.db"
    mem = ProjectMemory(db_path=mem_db)

    # 1. Project saving
    mem.save_project(
        project_id="proj-test-01",
        style="anime",
        target_duration=30.0,
        color_grade="anime",
        branding_name="KANKI.7",
        timeline_data={"test": True},
    )
    recent = mem.get_recent_projects(limit=5)
    assert len(recent) == 1
    assert recent[0]["project_id"] == "proj-test-01"

    # 2. Adaptive learning from user feedback
    adapt = AdaptiveEditor()
    res1 = adapt.process_feedback("The transitions are too slow")
    assert res1.recognized is True
    assert res1.parameter_adjusted == "transition_duration"

    res2 = adapt.process_feedback("Make it brighter and more vibrant")
    assert res2.recognized is True
