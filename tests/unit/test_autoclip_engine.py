"""JARVIS AutoClip Engine — Unit Test Suite
==========================================
Tests transcript slicing, sliding window management, viral highlight scoring,
timeline pause snapping, duplicate suppression, FFmpeg command generation,
compilation assembly, and full end-to-end AutoClip orchestration.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.brain.router import IntentRouter, IntentType
from video_editor.autoclip import (
    AutoClipEngine,
    AutoClipResult,
    ClipEvaluator,
    DurationProfile,
    RefinedClip,
    SlicingWindow,
    TimelineRefiner,
    TranscriptCue,
    TranscriptSlicer,
    VideoProcessor,
    CompilationBuilder,
    profile_from_duration,
    to_clock_time,
    to_seconds,
    to_srt_time,
)


# Sample SRT Data for Testing
SAMPLE_SRT = """1
00:00:01,000 --> 00:00:05,500
Welcome to the AI podcast. Today we discuss autonomous agents.

2
00:00:06,000 --> 00:00:12,000
Why do most developers fail when building autonomous systems?

3
00:00:12,500 --> 00:00:22,000
The truth is that they ignore deterministic guardrails and rely completely on black-box prompts.

4
00:00:22,500 --> 00:00:35,000
Here is the secret framework: combine offline-first architecture with strict invariant testing.

5
00:00:36,000 --> 00:00:50,000
This simple 10x shift increases reliability by over 90 percent. Always remember this critical lesson.

6
00:00:51,000 --> 00:01:05,000
Thank you for listening, subscribe for more technical deep dives.
"""


SAMPLE_VTT = """WEBVTT

00:01.000 --> 00:05.500
Hello and welcome everyone.

00:06.000 --> 00:15.000
What is the biggest mistake in machine learning engineering?

00:16.000 --> 00:30.000
It is shipping unverified models without comprehensive regression suites.
"""


def test_time_conversions():
    """Verify time conversions to seconds, SRT timestamp, and clock format."""
    assert to_seconds("00:01:30.500") == 90.5
    assert to_seconds("01:30,250") == 90.25
    assert to_seconds(45) == 45.0

    assert to_srt_time(90.5) == "00:01:30,500"
    assert to_srt_time(0.0) == "00:00:00,000"

    assert to_clock_time(65) == "1:05"
    assert to_clock_time(3665) == "1:01:05"


def test_transcript_parser_srt(tmp_path):
    """Verify parsing of standard SRT string and file input."""
    srt_file = tmp_path / "test.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")

    cues = TranscriptSlicer.parse_srt(srt_file)
    assert len(cues) == 6
    assert cues[0].id == 1
    assert cues[0].start_sec == 1.0
    assert cues[0].end_sec == 5.5
    assert "AI podcast" in cues[0].text

    assert cues[1].id == 2
    assert "Why do most developers fail" in cues[1].text

    # Test token-efficient row formatting
    rows = TranscriptSlicer.format_rows(cues[:2])
    assert "1|0:01|Welcome to the AI podcast" in rows
    assert "2|0:06|Why do most developers fail" in rows


def test_transcript_parser_vtt(tmp_path):
    """Verify WebVTT format parsing."""
    vtt_file = tmp_path / "test.vtt"
    vtt_file.write_text(SAMPLE_VTT, encoding="utf-8")

    cues = TranscriptSlicer.parse_vtt(vtt_file)
    assert len(cues) == 3
    assert cues[0].start_sec == 1.0
    assert "welcome everyone" in cues[0].text.lower()
    assert cues[1].start_sec == 6.0


def test_transcript_parser_raw_text():
    """Verify plain text splitting and proportional timing."""
    text = "First thought here. Second major point explained. Final conclusion."
    cues = TranscriptSlicer.parse_raw_text(text, total_duration_sec=30.0)
    assert len(cues) == 3
    assert cues[0].start_sec == 0.0
    assert cues[-1].end_sec == 30.0
    assert "conclusion" in cues[-1].text


def test_sliding_window_computation():
    """Verify overlapping window computation on long transcripts."""
    # Create 30 synthetic cues, 60 seconds each -> total 1800s (30m)
    cues = [
        TranscriptCue(id=i, start_sec=i * 60.0, end_sec=(i + 1) * 60.0, text=f"Cue sentence {i} discussing topics.")
        for i in range(30)
    ]
    windows = TranscriptSlicer.compute_windows(cues, window_sec=600.0, overlap_sec=120.0, max_chars=5000)
    assert len(windows) >= 3
    # Check overlap
    assert windows[0].end_sec > windows[1].start_sec


def test_duration_profile():
    """Verify duration profile tiers."""
    short_p = profile_from_duration(300.0)   # 5m
    assert short_p.tier == "short"
    assert short_p.min_clip_sec == 20.0

    med_p = profile_from_duration(1800.0)    # 30m
    assert med_p.tier == "medium"
    assert med_p.min_clip_sec == 30.0

    long_p = profile_from_duration(4000.0)   # > 1h
    assert long_p.tier == "long"
    assert long_p.min_clip_sec == 45.0


def test_timeline_refiner_snap():
    """Verify snapping raw timestamps to nearest cue boundaries."""
    cues = [
        TranscriptCue(id=1, start_sec=10.0, end_sec=20.0, text="First cue"),
        TranscriptCue(id=2, start_sec=20.5, end_sec=35.0, text="Second cue"),
    ]
    # Raw timestamp 11.2s should snap to 10.0s (within snap_window_sec=4.0s)
    snapped_start = TimelineRefiner.snap_to_cue_boundary(11.2, cues, prefer_start=True, max_dist=4.0)
    assert snapped_start == 10.0

    # Raw timestamp 34.2s should snap to 35.0s
    snapped_end = TimelineRefiner.snap_to_cue_boundary(34.2, cues, prefer_start=False, max_dist=4.0)
    assert snapped_end == 35.0


def test_timeline_refiner_deduplicate():
    """Verify deduplication of clips sharing > 50% overlap."""
    clip1 = RefinedClip(
        id=1, title="Top Moment A", reason="test", score=0.90,
        start_sec=10.0, end_sec=70.0, start_row=0, end_row=5, text_snippet="Sample"
    )
    # Clip 2 overlaps clip 1 by 50 seconds out of 60 seconds (83%)
    clip2 = RefinedClip(
        id=2, title="Top Moment B", reason="test", score=0.75,
        start_sec=20.0, end_sec=80.0, start_row=1, end_row=6, text_snippet="Sample"
    )
    # Clip 3 is completely independent
    clip3 = RefinedClip(
        id=3, title="Top Moment C", reason="test", score=0.85,
        start_sec=150.0, end_sec=210.0, start_row=10, end_row=15, text_snippet="Sample"
    )

    deduped = TimelineRefiner.deduplicate([clip1, clip2, clip3], overlap_threshold=0.5)
    # Clip 2 should be eliminated because it overlaps clip 1 and has lower score
    assert len(deduped) == 2
    titles = [c.title for c in deduped]
    assert "Top Moment A" in titles
    assert "Top Moment C" in titles
    assert "Top Moment B" not in titles


def test_clip_evaluator_heuristics():
    """Verify offline heuristic scoring detects viral hooks and questions."""
    cues = TranscriptSlicer.parse_srt(SAMPLE_SRT)
    evaluator = ClipEvaluator()

    profile = DurationProfile(tier="short", total_sec=65.0, min_clip_sec=15.0, target_clip_sec=(20.0, 45.0), max_clip_sec=60.0)
    clips = evaluator.evaluate_cues(cues, profile=profile, min_score=0.5)

    assert len(clips) >= 1
    top_clip = clips[0]
    assert top_clip.score > 0.5
    assert len(top_clip.title) > 0
    # Cues contain "Why do most developers fail" or "secret framework", should reflect in text snippet
    assert any(w in top_clip.text_snippet.lower() for w in ["why", "secret", "truth", "framework"])


def test_clip_evaluator_llm_parsing():
    """Verify LLM prompt response parsing."""
    mock_llm_response = json.dumps({
        "clips": [
            {
                "start": 1,
                "end": 3,
                "title": "Why Autonomous Systems Fail",
                "reason": "Addresses high-interest developer challenge with clear resolution",
                "score": 0.92
            }
        ]
    })
    mock_callable = MagicMock(return_value=mock_llm_response)
    evaluator = ClipEvaluator(llm_callable=mock_callable)

    cues = TranscriptSlicer.parse_srt(SAMPLE_SRT)
    profile = DurationProfile(tier="short", total_sec=65.0, min_clip_sec=10.0, target_clip_sec=(15.0, 40.0), max_clip_sec=60.0)
    clips = evaluator.evaluate_cues(cues, profile=profile, min_score=0.7)

    assert len(clips) >= 1
    assert "Why Autonomous Systems Fail" in [c.title for c in clips]


def test_video_processor_build_command(tmp_path):
    """Verify FFmpeg command line structure for vertical shorts."""
    proc = VideoProcessor(ffmpeg_bin="ffmpeg")
    input_v = tmp_path / "source.mp4"
    output_v = tmp_path / "clip.mp4"

    cmd = proc.build_extract_command(
        input_video=input_v,
        output_path=output_v,
        start_sec=10.5,
        duration_sec=30.0,
        vertical=True,
        watermark_text="JARVIS AI",
    )

    cmd_str = " ".join(cmd)
    assert "-ss 10.500" in cmd_str
    assert "-t 30.000" in cmd_str
    assert "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920" in cmd_str
    assert "-c:v libx264" in cmd_str
    assert "-movflags +faststart" in cmd_str
    assert "-avoid_negative_ts make_zero" in cmd_str


def test_compilation_builder_single_and_multi(tmp_path):
    """Verify compilation builder file logic."""
    clip1 = tmp_path / "c1.mp4"
    clip2 = tmp_path / "c2.mp4"
    clip1.write_bytes(b"dummy1")
    clip2.write_bytes(b"dummy2")

    builder = CompilationBuilder(ffmpeg_bin="ffmpeg")

    # Single clip copies directly
    single_out = tmp_path / "single_comp.mp4"
    assert builder.build_compilation([clip1], single_out) is True
    assert single_out.exists()


def test_autoclip_engine_end_to_end_dry_run(tmp_path):
    """Verify end-to-end AutoClip pipeline in dry-run mode."""
    # Create mock video and subtitle files
    vid_file = tmp_path / "interview.mp4"
    vid_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")  # MP4 header dummy
    srt_file = tmp_path / "interview.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")

    out_dir = tmp_path / "autoclip_output"

    engine = AutoClipEngine()
    result = engine.process_video(
        video_path=vid_file,
        srt_path=srt_file,
        output_dir=out_dir,
        vertical=True,
        min_score=0.5,
        max_clips=3,
        dry_run=True,
    )

    assert result.success is True
    assert len(result.clips) >= 1
    assert len(result.exported_clip_paths) >= 1
    assert result.metadata_path is not None
    assert Path(result.metadata_path).exists()

    # Read generated metadata
    meta_json = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
    assert meta_json["summary"]["clips_count"] == len(result.clips)
    assert "9:16 (vertical)" in meta_json["summary"]["aspect_ratio"]


def test_brain_router_autoclip_intent():
    """Verify IntentRouter routes autoclip queries to VideoEditingAgent."""
    router = IntentRouter()

    dec1 = router.route("/autoclip podcast.mp4")
    assert dec1.intent == IntentType.VIDEO_EDITING
    assert dec1.needs_agent is True
    assert dec1.agent_name == "VideoEditingAgent"
    assert "podcast.mp4" in dec1.extracted_query

    dec2 = router.route("extract viral clips from interview.mp4")
    assert dec2.intent == IntentType.VIDEO_EDITING
    assert dec2.agent_name == "VideoEditingAgent"

    dec3 = router.route("can you auto clip the highlights from my video.mp4?")
    assert dec3.intent == IntentType.VIDEO_EDITING
    assert dec3.agent_name == "VideoEditingAgent"


def test_is_supported_url():
    """Verify web video URL detection."""
    from video_editor.autoclip.url_ingest import is_supported_url

    assert is_supported_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
    assert is_supported_url("https://youtu.be/dQw4w9WgXcQ") is True
    assert is_supported_url("http://bilibili.com/video/BV1xx411c7mD") is True
    assert is_supported_url("C:\\Users\\harsh\\Videos\\podcast.mp4") is False
    assert is_supported_url("relative/path/video.mp4") is False
    assert is_supported_url("") is False


def test_autoclip_engine_with_mocked_url(tmp_path):
    """Verify AutoClipEngine downloads from URL via UrlIngest and clips video."""
    from video_editor.autoclip.url_ingest import UrlIngest, UrlDownloadResult

    mock_vid = tmp_path / "downloaded_interview.mp4"
    mock_vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    mock_srt = tmp_path / "downloaded_interview.srt"
    mock_srt.write_text(SAMPLE_SRT, encoding="utf-8")

    mock_result = UrlDownloadResult(
        success=True,
        video_path=mock_vid,
        srt_path=mock_srt,
        title="AI Engineering Podcast",
        duration_sec=65.0,
        error_message="",
    )

    with patch.object(UrlIngest, "download_media", return_value=mock_result) as mock_dl:
        engine = AutoClipEngine()
        res = engine.process_video(
            video_path="https://www.youtube.com/watch?v=sample123",
            output_dir=tmp_path / "url_autoclip_out",
            vertical=True,
            dry_run=True,
        )

        assert mock_dl.called
        assert res.success is True
        assert len(res.clips) >= 1
        assert len(res.exported_clip_paths) >= 1
        assert res.metadata_path is not None
        assert Path(res.metadata_path).exists()

