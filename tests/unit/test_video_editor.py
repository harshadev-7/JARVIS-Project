"""Unit test suite for J.A.R.V.I.S. Video Editing Automation (Phase 19)."""

import os
import struct
import tempfile
import wave
from pathlib import Path
import pytest

from video_editor.safety import MediaSafetyError, MediaSafetyGuard
from video_editor.inspector import VideoInspector, MediaMetadata
from video_editor.timeline import VideoTimeline, ClipSegment
from video_editor.scene_detector import SceneDetector, SceneChange
from video_editor.beat_sync import BeatSyncEngine, BeatPoint
from video_editor.subtitles import SubtitleManager, SubtitleEntry
from video_editor.effects import VideoEffectsEngine, SocialFormattingConfig
from video_editor.exporter import VideoExporter, ExportQuality, PROFILES
from video_editor.video_editing_agent import VideoEditingCoordinator
from core.brain.router import IntentRouter, IntentType
from core.orchestrator.agent_registry import VideoEditingAgent


@pytest.fixture
def temp_media_dir(tmp_path):
    """Fixture providing temporary media files and an isolated output dir."""
    out_dir = tmp_path / "output_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy video file
    dummy_video = tmp_path / "sample_test.mp4"
    dummy_video.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08free")

    # Create dummy 16-bit PCM WAV audio file with energy bursts
    dummy_wav = tmp_path / "sample_beat.wav"
    framerate = 8000
    n_samples = framerate * 2  # 2 seconds
    samples = []
    for i in range(n_samples):
        # Insert high energy burst every 0.5s (4000 samples)
        if (i % 4000) < 400:
            val = 20000
        else:
            val = 500
        samples.append(val)

    with wave.open(str(dummy_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    return {
        "out_dir": out_dir,
        "video": dummy_video,
        "wav": dummy_wav,
    }


def test_media_safety_guard(temp_media_dir):
    """Test safety guard enforcement of file formats and non-destructive operations."""
    guard = MediaSafetyGuard(output_dir=temp_media_dir["out_dir"])

    # 1. Valid extension
    p = guard.validate_file_path(temp_media_dir["video"])
    assert p.exists()

    # 2. Block unsupported extension
    with pytest.raises(ValueError):
        guard.validate_file_path("malicious_script.exe", must_exist=False)

    # 3. Block deletion without confirmation
    with pytest.raises(MediaSafetyError):
        guard.verify_action_permission(temp_media_dir["video"], is_deletion=True, user_confirmed=False)

    # Allowed when confirmed
    assert guard.verify_action_permission(temp_media_dir["video"], is_deletion=True, user_confirmed=True)

    # 4. Block in-place overwrite
    with pytest.raises(MediaSafetyError):
        guard.verify_action_permission(temp_media_dir["video"], destination_file=temp_media_dir["video"], user_confirmed=False)

    # 5. Non-destructive output filename collision avoidance
    out1 = guard.get_safe_output_path("result.mp4")
    out1.write_text("file 1")
    out2 = guard.get_safe_output_path("result.mp4")
    assert out1 != out2
    assert "result_1.mp4" in str(out2)


def test_video_inspector(temp_media_dir):
    """Test metadata inspection and fallback parsing."""
    inspector = VideoInspector()
    meta = inspector.inspect(temp_media_dir["video"])
    assert isinstance(meta, MediaMetadata)
    assert meta.filename == "sample_test.mp4"
    assert meta.width > 0
    assert meta.height > 0
    assert meta.aspect_ratio in ["16:9", "9:16"]


def test_video_timeline(temp_media_dir):
    """Test multi-clip timeline composition and FFmpeg concat script generation."""
    timeline = VideoTimeline(title="Test Reel")
    seg1 = timeline.add_clip(temp_media_dir["video"], start_sec=0.0, end_sec=5.0)
    seg2 = timeline.add_clip(temp_media_dir["video"], start_sec=10.0, end_sec=15.0, speed=2.0)

    assert timeline.total_clips == 2
    assert seg1.duration == 5.0
    assert seg2.duration == 2.5
    assert timeline.total_duration() == 7.5

    concat_file = timeline.generate_concat_file()
    assert concat_file.exists()
    content = concat_file.read_text(encoding="utf-8")
    assert "file " in content
    assert "inpoint 10.000" in content
    assert "outpoint 5.000" in content


def test_scene_detector(temp_media_dir):
    """Test scene cut detection and highlight ranking."""
    detector = SceneDetector()
    res = detector.detect_scenes(temp_media_dir["video"], min_scene_len_sec=1.0)
    assert res.total_scenes >= 1

    highlights = detector.find_highlights(temp_media_dir["video"], top_k=2)
    assert len(highlights) >= 1
    assert "start_sec" in highlights[0]
    assert "end_sec" in highlights[0]


def test_beat_sync_engine(temp_media_dir):
    """Test PCM WAV energy extraction and cut-to-beat alignment."""
    engine = BeatSyncEngine()
    beat_res = engine.analyze_audio(temp_media_dir["wav"], sensitivity=1.2)
    assert beat_res.duration_sec > 0
    assert beat_res.estimated_bpm > 0

    # Test beat snapping
    beats = [1.0, 2.0, 3.0, 4.0]
    raw_cuts = [1.08, 2.85]  # 1.08 is close to 1.0; 2.85 is close to 3.0
    aligned = engine.align_cuts_to_beats(raw_cuts, beats, tolerance_sec=0.2)
    assert 1.0 in aligned
    assert 3.0 in aligned

    # Test BPM cut grid
    grid = engine.generate_beat_grid(duration_sec=10.0, bpm=120.0, beats_per_cut=4)
    assert len(grid) == 4
    assert grid[0] == 2.0
    assert grid[1] == 4.0


def test_subtitle_manager(tmp_path):
    """Test SubRip timestamp formatting, serialization, parsing, and burning filter."""
    mgr = SubtitleManager()

    # 1. Timestamp formatting
    assert mgr.seconds_to_srt_time(0.0) == "00:00:00,000"
    assert mgr.seconds_to_srt_time(65.5) == "00:01:05,500"
    assert mgr.srt_time_to_seconds("00:01:05,500") == 65.5

    # 2. Serialize and parse SRT
    entries = [
        SubtitleEntry(index=1, start_sec=0.0, end_sec=2.5, text="Hello JARVIS"),
        SubtitleEntry(index=2, start_sec=3.0, end_sec=5.0, text="AI Video Automation"),
    ]
    srt_str = mgr.generate_srt(entries)
    assert "00:00:00,000 --> 00:00:02,500" in srt_str
    assert "Hello JARVIS" in srt_str

    parsed = mgr.parse_srt(srt_str)
    assert len(parsed) == 2
    assert parsed[0].text == "Hello JARVIS"
    assert parsed[1].start_sec == 3.0

    # 3. File saving
    srt_file = tmp_path / "test.srt"
    mgr.save_srt(entries, srt_file)
    assert srt_file.exists()

    # 4. Filter string with escaped path
    filter_str = mgr.build_ffmpeg_subtitles_filter(srt_file, font_size=28)
    assert "subtitles=" in filter_str
    assert "Fontsize=28" in filter_str


def test_video_effects_engine():
    """Test 9:16 vertical formatting, text overlays, and watermark filters."""
    engine = VideoEffectsEngine()

    # 1. Vertical 9:16 filter with blurred background
    blur_filter = engine.build_vertical_reels_filter(SocialFormattingConfig(blur_background=True))
    assert "boxblur=" in blur_filter
    assert "scale=1080:1920" in blur_filter

    # Center crop mode
    crop_filter = engine.build_vertical_reels_filter(SocialFormattingConfig(blur_background=False))
    assert "crop=ih*9/16" in crop_filter

    # 2. Drawtext filter
    text_filter = engine.build_text_overlay_filter("JARVIS AI Shorts", position="top_center", font_size=32)
    assert "drawtext=" in text_filter
    assert "JARVIS AI Shorts" in text_filter

    # 3. Watermark filter
    wm_filter = engine.build_watermark_filter("logo.png", position="top_right")
    assert "overlay=" in wm_filter


def test_video_exporter(temp_media_dir):
    """Test export quality profiles and script generation."""
    exporter = VideoExporter()
    res = exporter.export(
        input_file=temp_media_dir["video"],
        output_filename="test_out.mp4",
        quality=ExportQuality.DRAFT,
    )

    assert res.success is True
    assert res.quality_preset == "DRAFT"
    assert "test_out.mp4" in res.output_path
    assert "-preset ultrafast" in res.command_str
    assert "-threads 2" in res.command_str


@pytest.mark.asyncio
async def test_video_editing_agent_and_coordinator(temp_media_dir):
    """Test high-level coordinator and VideoEditingAgent execution."""
    coord = VideoEditingCoordinator()
    coord.safety.output_dir = temp_media_dir["out_dir"]

    # Reel generation test
    res = coord.create_short_or_reel(
        video_path=temp_media_dir["video"],
        output_name="my_short.mp4",
        watermark_text="JARVIS AI",
        subtitles=["Turn your clips", "into AI reels!"],
    )
    assert res.success is True
    assert "my_short.mp4" in res.output_path

    # Trim test
    trim_res = coord.trim_clip(temp_media_dir["video"], start_sec=0.0, end_sec=3.0)
    assert trim_res.success is True

    # VideoEditingAgent execution
    agent = VideoEditingAgent()
    task_res = await agent.execute(
        f"make a 9:16 vertical reel from '{temp_media_dir['video']}'",
        context={"video_path": str(temp_media_dir["video"])},
    )
    assert task_res.success is True
    assert "Created 9:16 vertical reel" in task_res.summary


def test_router_video_editing_intent():
    """Test IntentRouter classification of video editing queries."""
    router = IntentRouter()

    queries = [
        "/video edit clip.mp4",
        "/reel create 9:16 short",
        "JARVIS, edit video and trim out the first 5 seconds",
        "can you make a reel from this clip",
        "burn subtitles into my video",
    ]

    for q in queries:
        decision = router.route(q)
        assert decision.intent == IntentType.VIDEO_EDITING
        assert decision.needs_agent is True
        assert decision.agent_name == "VideoEditingAgent"


@pytest.mark.asyncio
async def test_autonomous_video_creator_without_source_sample(tmp_path):
    """Test generating a 100% autonomous video with auto-curated viral trending content."""
    from video_editor.autonomous_creator import TrendingTopicCurator, AutonomousVideoCreator, ViralStoryBrief

    curator = TrendingTopicCurator()
    # 1. Autonomous topic selection
    brief = await curator.pick_or_research_topic()
    assert isinstance(brief, ViralStoryBrief)
    assert len(brief.topic) > 0
    assert len(brief.hook) > 0
    assert len(brief.bullet_points) == 3
    assert len(brief.subtitle_entries) == 5

    # 2. Autonomous video synthesis to custom save folder
    custom_out = tmp_path / "custom_reels"
    creator = AutonomousVideoCreator()
    result = await creator.create_autonomous_video(
        topic="Autonomous AI Agents",
        save_directory=custom_out,
        watermark_text="JARVIS AI",
    )
    assert result.success is True
    assert custom_out.exists()
    assert "Autonomous Video" in result.message

    # 3. Test VideoEditingAgent invocation without any source file
    agent = VideoEditingAgent()
    task_res = await agent.execute(
        "create a viral video on trending tech",
        context={"save_dir": str(custom_out)},
    )
    assert task_res.success is True
    assert "Autonomous Video" in task_res.summary


@pytest.mark.asyncio
async def test_video_playback_routing_and_execution(tmp_path):
    """Test that 'play that video than' is routed to VideoEditingAgent and executed."""
    router = IntentRouter()
    # 1. Test routing for conversational follow-ups and typos
    queries = [
        "play that video than",
        "play that video then",
        "open the video",
        "play the reel",
        "watch that video",
    ]
    for q in queries:
        dec = router.route(q)
        assert dec.intent == IntentType.VIDEO_EDITING
        assert dec.needs_agent is True
        assert dec.agent_name == "VideoEditingAgent"

    # 2. Test VideoEditingAgent playback execution with synthesized audio/script
    dummy_wav = tmp_path / "test_voice.wav"
    dummy_wav.write_bytes(b"RIFF" + b"\x00" * 40)
    dummy_bat = tmp_path / "render_test.bat"
    dummy_bat.write_text("@echo off", encoding="utf-8")

    agent = VideoEditingAgent()
    play_res = await agent.execute(
        "play that video than",
        context={
            "last_created_media": {
                "audio": str(dummy_wav),
                "script": str(dummy_bat),
            }
        },
    )
    assert play_res.success is True
    assert "Playing synthesized voiceover audio track" in play_res.summary

