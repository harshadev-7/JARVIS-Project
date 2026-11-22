"""J.A.R.V.I.S. Visual Intelligence Autonomous Agent (Phase 20).

High-level coordinator unifying all Computer Vision capabilities:
- Image classification via local TinyVisionNet
- Object and focal subject detection
- Scene understanding and lighting diagnostics
- Local Optical Character Recognition (OCR)
- Visual vector embedding and similarity search
- Phase 19 video auto-framing integration
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logger import log
from core.orchestrator.base_agent import AgentTaskResult, BaseAgent
from vision.classifier import image_classifier
from vision.frame_analyzer import frame_analyzer
from vision.image_loader import image_loader
from vision.object_detector import object_detector
from vision.ocr_engine import ocr_engine
from vision.scene_understanding import scene_analyzer
from vision.smart_framer import smart_auto_framer
from vision.visual_search import visual_search_engine


class VisionAgent(BaseAgent):
    """Specialized autonomous agent for local computer vision and visual intelligence."""

    def __init__(self):
        super().__init__(
            name="VisionAgent",
            role="Local visual intelligence: Image classification, object detection, OCR, scene understanding, and smart auto-framing.",
            capabilities=[
                "image_classification",
                "object_detection",
                "scene_understanding",
                "ocr_text_extraction",
                "visual_similarity_search",
                "video_frame_analysis",
                "smart_auto_framing",
            ],
            required_autonomy_level=2,
        )

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentTaskResult:
        """Execute vision tasks across all image and video frame workflows."""
        start_time = time.perf_counter()
        ctx = context or {}
        action = ctx.get("action", "auto")
        image_path = ctx.get("image_path") or ctx.get("file_path")

        # Check if task specifies an explicit path
        if not image_path:
            m = re.search(r"['\"]?([A-Za-z]:\\[^'\"\n]+|\b[\w\-_\.]+\.(?:png|jpg|jpeg|bmp|webp|ppm))\b", task)
            if m:
                image_path = m.group(1)

        # If no explicit image file, provide capability status and telemetry
        if not image_path:
            duration = time.perf_counter() - start_time
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"vision_info_{int(time.time())}",
                success=True,
                data={
                    "model": "TinyVisionNet-CPU",
                    "parameters": image_classifier.model.count_parameters(),
                    "capabilities": self.capabilities,
                    "offline": True,
                },
                summary="VisionAgent online: Ready for image classification, OCR, object detection, and visual search.",
                duration_sec=duration,
            )

        p = Path(image_path).resolve()
        if not p.exists():
            duration = time.perf_counter() - start_time
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"vision_err_{int(time.time())}",
                success=False,
                error=f"Target image file not found: {p}",
                summary=f"Visual operation failed: Image '{p.name}' does not exist on disk.",
                duration_sec=duration,
            )

        lower_task = task.lower()
        # Strip filepath from task text for keyword intent extraction
        clean_task = re.sub(r"['\"]?[A-Za-z]:\\[^'\"\n]+['\"]?", "", lower_task)
        clean_task = re.sub(r"['\"]?[\w\-_\.]+\.(?:png|jpg|jpeg|bmp|webp|ppm)['\"]?", "", clean_task)

        # Action 1: OCR / Text Extraction
        if action == "ocr" or any(w in clean_task for w in ["ocr", "extract text", "read text", "scan text"]):
            ocr_res = ocr_engine.extract_text(p)
            duration = time.perf_counter() - start_time
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"vision_ocr_{int(time.time())}",
                success=True,
                data={"full_text": ocr_res.full_text, "regions_count": len(ocr_res.regions), "engine": ocr_res.engine_used},
                summary=f"Extracted {ocr_res.total_words} words via local OCR:\n{ocr_res.full_text[:300]}...",
                duration_sec=duration,
            )

        # Action 2: Object Detection / Bounding boxes
        elif action == "detect" or any(w in clean_task for w in ["detect", "bounding box", "find objects", "detect faces", "locate object"]):
            det_res = object_detector.detect_objects(p)
            duration = time.perf_counter() - start_time
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"vision_det_{int(time.time())}",
                success=True,
                data={
                    "total_objects": det_res.total_objects,
                    "subject_center": det_res.primary_subject_center,
                    "objects": [b.__dict__ for b in det_res.objects],
                },
                summary=f"Detected {det_res.total_objects} focal visual regions. Primary subject center: {det_res.primary_subject_center}.",
                duration_sec=duration,
            )

        # Action 3: Scene Understanding / Lighting Analysis
        elif action == "scene" or any(w in clean_task for w in ["scene", "lighting", "quality", "blur", "sharpness"]):
            scene_res = scene_analyzer.analyze_scene(p)
            duration = time.perf_counter() - start_time
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"vision_scene_{int(time.time())}",
                success=True,
                data=scene_res.__dict__,
                summary=f"Scene Diagnostics: Lighting={scene_res.lighting_condition}, Sharpness={scene_res.sharpness:.1f}, Dominant RGB={scene_res.dominant_color_rgb}.",
                duration_sec=duration,
            )

        # Action 4: Visual Similarity Search & Indexing
        elif action == "search" or any(w in clean_task for w in ["similar", "reverse image", "search visual"]):
            matches = visual_search_engine.search_similar(p, top_k=3)
            duration = time.perf_counter() - start_time
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"vision_search_{int(time.time())}",
                success=True,
                data={"matches": [m.__dict__ for m in matches]},
                summary=f"Visual Similarity Search: Found {len(matches)} matching records in local database.",
                duration_sec=duration,
            )

        # Action 5: Smart Auto-Framing for Vertical Video
        elif action == "auto_frame" or any(w in clean_task for w in ["auto frame", "smart frame", "crop 9:16", "crop vertical", "reel framing"]):
            x_coord = smart_auto_framer.calculate_pan_coordinate(p)
            duration = time.perf_counter() - start_time
            return AgentTaskResult(
                agent_name=self.name,
                task_id=f"vision_frame_{int(time.time())}",
                success=True,
                data={"pan_x_offset": x_coord},
                summary=f"Smart Auto-Framing: Subject center pan offset calculated at x={x_coord}px for vertical 9:16 formatting.",
                duration_sec=duration,
            )

        # Default: General Image Classification
        cls_res = image_classifier.classify(p)
        meta = image_loader.inspect_file(p)
        duration = time.perf_counter() - start_time

        return AgentTaskResult(
            agent_name=self.name,
            task_id=f"vision_cls_{int(time.time())}",
            success=True,
            data={
                "label": cls_res.primary_label,
                "confidence": cls_res.confidence,
                "top_k": cls_res.top_k,
                "dimensions": f"{meta.width}x{meta.height}",
                "format": meta.format,
            },
            summary=f"Visual Classification: '{cls_res.primary_label}' ({cls_res.confidence*100:.1f}% confidence, {cls_res.latency_ms:.1f}ms latency).",
            duration_sec=duration,
        )


vision_agent = VisionAgent()

__all__ = ["VisionAgent", "vision_agent"]
