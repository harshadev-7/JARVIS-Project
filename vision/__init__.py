"""J.A.R.V.I.S. Computer Vision Subsystem (Phase 20).

Provides local visual intelligence:
- Image inspection, tensor parsing, and memory safety
- TinyVisionNet local convolutional neural network
- Object and visual saliency detection
- Scene understanding and lighting diagnostics
- Local OCR and text extraction
- Visual vector embeddings and similarity search
- Dynamic 9:16 Subject Auto-Framing (Phase 19 integration)
- Specialized VisionAgent
"""

from vision.image_loader import ImageMetadata, ImageLoader, image_loader
from vision.classifier import ClassificationResult, TinyVisionNet, ImageClassifier, image_classifier
from vision.object_detector import BoundingBox, DetectionResult, ObjectDetector, object_detector
from vision.scene_understanding import SceneReport, SceneAnalyzer, scene_analyzer
from vision.frame_analyzer import FrameAnalysisPoint, VideoVisualSummary, VideoFrameAnalyzer, frame_analyzer
from vision.ocr_engine import TextRegion, OCRResult, OCREngine, ocr_engine
from vision.visual_search import VisualSearchResult, VisualSearchEngine, visual_search_engine
from vision.smart_framer import AutoFrameConfig, SmartAutoFramer, smart_auto_framer
from vision.vision_agent import VisionAgent, vision_agent

__all__ = [
    "ImageMetadata",
    "ImageLoader",
    "image_loader",
    "ClassificationResult",
    "TinyVisionNet",
    "ImageClassifier",
    "image_classifier",
    "BoundingBox",
    "DetectionResult",
    "ObjectDetector",
    "object_detector",
    "SceneReport",
    "SceneAnalyzer",
    "scene_analyzer",
    "FrameAnalysisPoint",
    "VideoVisualSummary",
    "VideoFrameAnalyzer",
    "frame_analyzer",
    "TextRegion",
    "OCRResult",
    "OCREngine",
    "ocr_engine",
    "VisualSearchResult",
    "VisualSearchEngine",
    "visual_search_engine",
    "AutoFrameConfig",
    "SmartAutoFramer",
    "smart_auto_framer",
    "VisionAgent",
    "vision_agent",
]
