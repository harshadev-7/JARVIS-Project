"""
JARVIS Video Editing Engine V2.0 — Comprehensive Test Suite
===========================================================
Tests V2.0 enhancements:
- Copyright Protection Gatekeeper (Goku, Dragon Ball, Anime, Marvel)
- Autonomous Media Discovery Engine (Wikimedia Commons & Internet Archive)
- Mandatory Footage Content Validation in Quality Control (Rejects fake color canvases)
- CapCut-Style Effects Engine (Velocity curves, impact camera shake, RGB split)
- 6-Track MultiTrackTimeline NLE Model (V1, V2, V3, T1, A1, A2)
- Autonomous /reel execution with real footage validation
"""

import pytest
import subprocess
from pathlib import Path

from video_editor.media_discovery import MediaDiscoveryEngine, DiscoveredMediaItem, MediaCollectionReport
from video_editor.capcut_engine import CapCutEngine, capcut_engine
from video_editor.multitrack_timeline import MultiTrackTimeline, create_multitrack_timeline
from video_editor.quality_control import QualityControlEngine, quality_control_engine
from video_editor.reel_controller import ReelController
from video_editor.workspace import workspace_manager


@pytest.fixture
def test_real_video(tmp_path):
    """Generate a high-variance synthetic video clip with motion."""
    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    out_video = tmp_path / "real_footage_sample.mp4"

    # Multi-color dynamic moving test pattern
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


@pytest.fixture
def test_fake_canvas_video(tmp_path):
    """Generate a flat dark color canvas video with text overlay to test QC rejection."""
    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    out_video = tmp_path / "fake_canvas_sample.mp4"

    cmd = [
        ffmpeg_bin, "-y",
        "-f", "lavfi", "-i", "color=c=0x0b0f19:s=1080x1920:d=3:r=30",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-t", "3",
        str(out_video),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out_video


# ── 1. Copyright Protection Gatekeeper Tests ─────────────────────────────────

def test_copyright_protection_gatekeeper():
    """Verify that copyrighted anime/commercial queries are flagged."""
    discovery = MediaDiscoveryEngine()

    is_prot, msg = discovery.is_copyright_protected("Goku Ultra Instinct edit")
    assert is_prot is True
    assert "Dragon Ball" in msg or "Goku" in msg
    assert "media/input/" in msg

    is_prot_naruto, msg_naruto = discovery.is_copyright_protected("Naruto vs Sasuke fight")
    assert is_prot_naruto is True

    # Generic or public domain queries should NOT be blocked
    is_prot_cinematic, _ = discovery.is_copyright_protected("cinematic mountains 30s")
    assert is_prot_cinematic is False

    is_prot_cars, _ = discovery.is_copyright_protected("classic vintage cars 4k")
    assert is_prot_cars is False


# ── 2. Quality Control Footage Content Validation Tests ──────────────────────

def test_quality_control_rejects_fake_canvas(test_fake_canvas_video):
    """Verify QC rejects flat solid background videos as missing real footage."""
    qc = QualityControlEngine()
    rep = qc.inspect_export(test_fake_canvas_video, expected_duration_sec=3.0)

    assert rep.has_real_footage is False
    assert rep.passed is False
    assert any("Mandatory footage check failed" in issue for issue in rep.issues)


def test_quality_control_accepts_real_footage(test_real_video):
    """Verify QC accepts videos containing actual visual variance and motion."""
    qc = QualityControlEngine()
    rep = qc.inspect_export(test_real_video, expected_duration_sec=3.0)

    assert rep.has_real_footage is True
    assert rep.passed is True
    assert rep.spatial_variance >= 10.0


# ── 3. CapCut-Style Effects Engine Tests ──────────────────────────────────────

def test_capcut_engine_effects():
    """Verify CapCut-grade velocity, shake, glitch, and grade expressions."""
    engine = CapCutEngine()

    # Velocity ramping
    v_slow, a_slow = engine.build_velocity_ramp_filter(ramp_type="impact_slowmo")
    assert "setpts=" in v_slow
    assert "between(T,1.2,2.4)" in v_slow

    # Impact camera shake
    shake_filter = engine.build_camera_shake_filter(intensity=0.7)
    assert "crop=" in shake_filter
    assert "sin(t*" in shake_filter
    assert "cos(t*" in shake_filter

    # Chromatic aberration RGB glitch
    glitch_filter = engine.build_rgb_split_glitch_filter(intensity=0.8)
    assert "rgbashift=" in glitch_filter

    # Flash bloom
    flash_filter = engine.build_flash_bloom_filter(start_sec=1.5, duration_sec=0.3)
    assert "fade=t=in:st=1.5" in flash_filter

    # Ultra Instinct & Phonk grades
    ui_grade = engine.build_ultra_instinct_grade_filter()
    assert "colorbalance=" in ui_grade
    assert "unsharp=" in ui_grade

    phonk_grade = engine.build_phonk_aggressive_grade_filter()
    assert "noise=" in phonk_grade


# ── 4. Multi-Track Timeline Model Tests ───────────────────────────────────────

def test_multitrack_timeline_6_tracks(tmp_path):
    """Verify 6-track NLE timeline construction and serialization."""
    mt = create_multitrack_timeline(
        project_id="proj-v2-test",
        title="Ultra Instinct Edit",
        duration_sec=30.0,
        aspect_ratio="9:16",
    )

    # Add items to all 6 tracks
    mt.add_clip("v1", source_path="clip1.mp4", timeline_in=0.0, duration=10.0)
    mt.add_clip("v2", source_path="broll_overlay.mp4", timeline_in=5.0, duration=3.0)
    mt.add_clip("v3", source_path="adjustment_layer", timeline_in=0.0, duration=30.0, effects=["ui_grade", "rgb_split"])
    mt.add_clip("t1", source_path="THE GOAT NEVER FALLS", timeline_in=0.0, duration=3.0)
    mt.add_clip("a1", source_path="powerful_phonk.mp3", timeline_in=0.0, duration=30.0)
    mt.add_clip("a2", source_path="impact_sfx.wav", timeline_in=10.0, duration=1.5)

    assert len(mt.tracks["v1"]) == 1
    assert len(mt.tracks["v2"]) == 1
    assert len(mt.tracks["v3"]) == 1
    assert len(mt.tracks["t1"]) == 1
    assert len(mt.tracks["a1"]) == 1
    assert len(mt.tracks["a2"]) == 1

    json_path = tmp_path / "test_multitrack.json"
    saved_file = mt.save_project_json(output_file=json_path)
    assert saved_file.exists()
    assert "proj-v2-test" in saved_file.read_text(encoding="utf-8")


# ── 5. /reel Controller V2.0 Integration Tests ───────────────────────────────

@pytest.mark.asyncio
async def test_reel_controller_refuses_fake_video_when_copyrighted(monkeypatch):
    """Verify that requesting copyrighted anime without footage returns an alert instead of a fake video."""
    monkeypatch.setattr(workspace_manager, "list_input_videos", lambda: [])
    ctrl = ReelController()
    result = await ctrl.execute_command("/reel Goku Ultra Instinct 30s")

    assert "Copyright Protection Alert" in result
    assert "media/input/" in result
    assert "cannot legally provide" in result
