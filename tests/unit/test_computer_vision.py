"""Unit test suite for J.A.R.V.I.S. Computer Vision Subsystem (Phase 20)."""

import os
import tempfile
from pathlib import Path
import pytest
import torch

from vision.image_loader import ImageLoader, ImageMetadata, image_loader
from vision.classifier import TinyVisionNet, ImageClassifier, image_classifier, ClassificationResult
from vision.object_detector import ObjectDetector, BoundingBox, DetectionResult, object_detector
from vision.scene_understanding import SceneAnalyzer, SceneReport, scene_analyzer
from vision.frame_analyzer import VideoFrameAnalyzer, VideoVisualSummary, frame_analyzer
from vision.ocr_engine import OCREngine, TextRegion, OCRResult, ocr_engine
from vision.visual_search import VisualSearchEngine, VisualSearchResult
from vision.smart_framer import SmartAutoFramer, AutoFrameConfig, smart_auto_framer
from vision.vision_agent import VisionAgent, vision_agent
from core.brain.router import IntentRouter, IntentType
from core.orchestrator.agent_registry import agent_registry


@pytest.fixture
def temp_vision_data(tmp_path):
    """Fixture providing a synthetic image and isolated sqlite database."""
    img_path = tmp_path / "test_frame.bmp"
    image_loader.create_synthetic_bmp(img_path, width=256, height=144, color_rgb=(0, 200, 255))

    db_path = tmp_path / "test_vision.db"
    return {
        "img": img_path,
        "db": db_path,
    }


def test_image_loader_inspection_and_tensor(temp_vision_data):
    """Test image header parsing, dimension extraction, and PyTorch tensor loading."""
    meta = image_loader.inspect_file(temp_vision_data["img"])
    assert isinstance(meta, ImageMetadata)
    assert meta.width == 256
    assert meta.height == 144
    assert meta.format == "BMP"
    assert meta.channels in (3, 4)
    assert meta.is_vertical is False

    tensor = image_loader.load_tensor(temp_vision_data["img"], target_size=(224, 224), normalize=True)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_tiny_vision_net_architecture():
    """Verify TinyVisionNet parameter budget and CPU forward pass."""
    model = TinyVisionNet(num_classes=10)
    params = model.count_parameters()
    # Ensure parameter count stays within lightweight CPU budget (<250k)
    assert 50_000 < params < 250_000

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (1, 10)


def test_image_classifier_inference(temp_vision_data):
    """Test classification execution, latency measurement, and top-k output."""
    res = image_classifier.classify(temp_vision_data["img"], top_k=3)
    assert isinstance(res, ClassificationResult)
    assert res.primary_label in TinyVisionNet.CLASSES
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.top_k) <= 3
    assert res.latency_ms >= 0.0


def test_object_and_saliency_detection(temp_vision_data):
    """Test gradient edge energy computation and subject center localization."""
    res = object_detector.detect_objects(temp_vision_data["img"], max_objects=3)
    assert isinstance(res, DetectionResult)
    assert res.total_objects >= 1
    assert len(res.objects) >= 1

    cx, cy = res.primary_subject_center
    assert 0.0 <= cx <= 1.0
    assert 0.0 <= cy <= 1.0

    primary_box = res.objects[0]
    assert primary_box.width > 0
    assert primary_box.height > 0
    assert primary_box.area > 0


def test_scene_understanding(temp_vision_data):
    """Test scene lighting, sharpness, and dominant color calculation."""
    report = scene_analyzer.analyze_scene(temp_vision_data["img"])
    assert isinstance(report, SceneReport)
    assert 0.0 <= report.brightness <= 1.0
    assert report.contrast >= 0.0
    assert report.sharpness >= 0.0
    assert report.lighting_condition in ("underexposed", "balanced", "overexposed")
    assert len(report.dominant_color_rgb) == 3


def test_video_frame_analyzer(temp_vision_data):
    """Test temporal activity scoring and peak motion interval detection."""
    # Create dummy video path
    dummy_vid = temp_vision_data["img"].parent / "mock_video.mp4"
    dummy_vid.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08free")

    summary = frame_analyzer.analyze_video_frames(dummy_vid, sample_rate_sec=1.0, max_samples=5)
    assert isinstance(summary, VideoVisualSummary)
    assert summary.total_frames_sampled > 0
    assert summary.average_motion >= 0.0


def test_local_ocr_engine(temp_vision_data):
    """Test text region extraction and OCR transcript generation."""
    res = ocr_engine.extract_text(temp_vision_data["img"])
    assert isinstance(res, OCRResult)
    assert res.full_text != ""
    assert len(res.regions) > 0
    assert res.total_words > 0


def test_visual_search_and_vector_indexing(temp_vision_data):
    """Test 64-d visual embedding calculation, SQLite storage, and similarity ranking."""
    search_eng = VisualSearchEngine(db_path=temp_vision_data["db"])

    # 1. Compute embedding
    vec = search_eng.compute_embedding(temp_vision_data["img"])
    assert len(vec) == 64
    # Check unit normalization: sum of squares ~= 1.0
    sq_norm = sum(v * v for v in vec)
    assert 0.95 <= sq_norm <= 1.05

    # 2. Index into SQLite
    indexed = search_eng.index_image(temp_vision_data["img"], description="Sample Cyan Test Card")
    assert indexed is True

    # 3. Reverse image lookup
    matches = search_eng.search_similar(temp_vision_data["img"], top_k=2)
    assert len(matches) == 1
    assert matches[0].similarity_score >= 0.99  # Self-match should be ~1.0


def test_smart_auto_framer(temp_vision_data):
    """Test subject-aware pan offset calculation and 9:16 FFmpeg filter generation."""
    framer = SmartAutoFramer()
    x_offset = framer.calculate_pan_coordinate(temp_vision_data["img"], source_width=1920, source_height=1080)
    assert 0 <= x_offset <= (1920 - 607)

    crop_filter = framer.build_dynamic_crop_filter([(0.0, x_offset)], source_width=1920, source_height=1080)
    assert "crop=607:1080:" in crop_filter
    assert "scale=1080:1920" in crop_filter


@pytest.mark.asyncio
async def test_vision_agent_execution(temp_vision_data):
    """Test VisionAgent dispatching across classification, OCR, detection, and info."""
    agent = VisionAgent()

    # 1. Telemetry info mode (no image path)
    info_res = await agent.execute("JARVIS, what are your computer vision capabilities?")
    assert info_res.success is True
    assert "TinyVisionNet-CPU" in str(info_res.data)

    # 2. General classification
    cls_task = await agent.execute(f"classify this picture: '{temp_vision_data['img']}'")
    assert cls_task.success is True
    assert "Visual Classification:" in cls_task.summary

    # 3. Object detection action
    det_task = await agent.execute("detect objects", context={"image_path": str(temp_vision_data["img"]), "action": "detect"})
    assert det_task.success is True
    assert "Detected" in det_task.summary

    # 4. OCR action
    ocr_task = await agent.execute("extract text", context={"image_path": str(temp_vision_data["img"]), "action": "ocr"})
    assert ocr_task.success is True
    assert "Extracted" in ocr_task.summary


def test_agent_registry_and_router_integration():
    """Verify VisionAgent is in agent_registry and routed by IntentRouter."""
    # 1. Agent registry
    reg_agent = agent_registry.get_agent("vision")
    assert reg_agent is not None
    assert reg_agent.name == "VisionAgent"

    # 2. Intent router
    router = IntentRouter()
    queries = [
        "/vision inspect diagram.png",
        "/ocr extract text from screenshot.bmp",
        "JARVIS, analyze this image and detect objects",
        "run visual search to find similar image",
    ]
    for q in queries:
        decision = router.route(q)
        assert decision.intent == IntentType.COMPUTER_VISION
        assert decision.needs_agent is True
        assert decision.agent_name == "VisionAgent"
