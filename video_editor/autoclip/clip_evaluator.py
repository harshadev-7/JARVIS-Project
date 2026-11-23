"""JARVIS AutoClip Engine — Viral Clip Evaluator & Highlight Finder
=================================================================
Scores and identifies candidate video highlights from transcript cues.
Employs a hybrid evaluation strategy:
1. Fast, offline rule-based heuristic scoring (hooks, Q&A pairs, emotion, density).
2. Optional LLM / local transformer evaluation using structured JSON prompting.
3. Clean fallback guaranteeing 100% offline-first reliability.

Inspired by AutoClip's one-pass clip finder and viral scoring metrics.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from video_editor.autoclip.transcript_slicer import (
    SlicingWindow,
    TranscriptCue,
    TranscriptSlicer,
    to_clock_time,
)
from video_editor.autoclip.timeline_refiner import (
    DurationProfile,
    RefinedClip,
    TimelineRefiner,
    profile_from_duration,
)

logger = logging.getLogger(__name__)

# Core Viral Markers & Keywords
HOOK_PATTERNS = [
    r"\b(how to|why you|the secret|the truth|never do|biggest mistake|must know|game changer|unbelievable)\b",
    r"\b(what happens when|the real problem|nobody talks about|i discovered|here is why|stop doing)\b",
    r"\b(key takeaway|in summary|the solution is|critical point|always remember|first step)\b",
    r"\b(10x|100%|billion|million|zero to|from scratch|hack|framework|system)\b",
]

EMOTION_KEYWORDS = {
    "amazing", "shocking", "insane", "incredible", "essential", "crucial", "huge",
    "fascinating", "ridiculous", "powerful", "genius", "failure", "danger", "secret",
    "transform", "revolution", "breakthrough", "worst", "best", "truth", "lie"
}

QUESTION_STARTERS = re.compile(r"^(what|why|how|who|where|when|can|is|are|do|does|did|could|should|would)\b", re.IGNORECASE)


@dataclass
class ScoredSegment:
    """A scored candidate highlight segment before boundary snapping."""
    start_row: int
    end_row: int
    title: str
    reason: str
    score: float
    hook_score: float
    density_score: float
    completeness_score: float
    tags: List[str]


class ClipEvaluator:
    """Evaluates transcript windows to discover the highest-potential viral clips."""

    AUTOCLIP_PROMPT_TEMPLATE = (
        "You are an expert short-form video editor and viral content strategist.\n"
        "Analyze the following transcript cues (formatted as id|time|text) and extract "
        "the best standalone video clips suitable for TikTok, YouTube Shorts, and Instagram Reels.\n\n"
        "CRITERIA:\n"
        "- Hook: Strong opening statement or question that grabs attention in the first 3 seconds.\n"
        "- Information Value: Concrete insights, unique perspective, or actionable knowledge.\n"
        "- Standalone Completeness: Must form a complete thought or complete Q&A.\n"
        "- Target Duration: {min_sec}s to {max_sec}s.\n\n"
        "TRANSCRIPT CUES:\n"
        "{rows}\n\n"
        "Output ONLY valid JSON in this exact structure:\n"
        '{{"clips": [{{"start": 0, "end": 10, "title": "Viral Hook Title", "reason": "Why this moment will perform well", "score": 0.85}}]}}'
    )

    def __init__(self, llm_callable: Optional[Callable[[str], str]] = None):
        """Optional llm_callable takes a prompt string and returns model output text."""
        self.llm_callable = llm_callable

    def evaluate_cues(
        self,
        cues: List[TranscriptCue],
        profile: Optional[DurationProfile] = None,
        min_score: float = 0.65,
    ) -> List[RefinedClip]:
        """Runs end-to-end highlight finding across all cues using windowing and hybrid scoring."""
        if not cues:
            return []

        total_dur = cues[-1].end_sec - cues[0].start_sec
        active_profile = profile or profile_from_duration(total_dur)
        windows = TranscriptSlicer.compute_windows(cues)

        candidate_dicts: List[Dict[str, Any]] = []

        # Analyze each sliding window
        for win in windows:
            window_cues = cues[win.first_idx : win.last_idx + 1]
            scored_candidates: List[ScoredSegment] = []

            # 1. Try LLM if callable is provided
            if self.llm_callable:
                try:
                    scored_candidates = self._evaluate_with_llm(window_cues, active_profile, win.first_idx)
                except Exception as err:
                    logger.warning("LLM evaluation failed, falling back to heuristic scoring: %s", err)
                    scored_candidates = []

            # 2. If no LLM results, use deterministic offline heuristic scorer
            if not scored_candidates:
                scored_candidates = self._evaluate_with_heuristics(window_cues, active_profile, win.first_idx)

            # Convert to candidate dicts
            for sc in scored_candidates:
                if sc.score >= min_score:
                    candidate_dicts.append({
                        "start_row": sc.start_row,
                        "end_row": sc.end_row,
                        "title": sc.title,
                        "reason": sc.reason,
                        "score": sc.score,
                        "tags": sc.tags,
                    })

        # Refine, snap, and deduplicate
        refined_clips = TimelineRefiner.refine_clips(candidate_dicts, cues, active_profile)

        # Fallback if no clips exceeded min_score: extract top 1 moment automatically
        if not refined_clips and candidate_dicts:
            candidate_dicts.sort(key=lambda x: x["score"], reverse=True)
            refined_clips = TimelineRefiner.refine_clips(candidate_dicts[:active_profile.min_keep], cues, active_profile)
        elif not refined_clips and cues:
            # Synthetic single highlight clip covering the most engaging portion
            top_span = self._extract_default_span(cues, active_profile)
            refined_clips = TimelineRefiner.refine_clips([top_span], cues, active_profile)

        return refined_clips

    def _evaluate_with_heuristics(
        self,
        window_cues: List[TranscriptCue],
        profile: DurationProfile,
        base_offset: int,
    ) -> List[ScoredSegment]:
        """Deterministic heuristic viral scoring across candidate spans."""
        candidates: List[ScoredSegment] = []
        n = len(window_cues)
        if n == 0:
            return []

        # Generate candidate (start_idx, end_idx) spans that fit target duration
        target_min, target_max = profile.target_clip_sec
        step = max(1, n // 15)  # Sample every few cues to avoid $O(N^2)$ overhead

        for i in range(0, n, step):
            start_cue = window_cues[i]
            for j in range(i + 1, min(i + 40, n)):
                end_cue = window_cues[j]
                span_dur = end_cue.end_sec - start_cue.start_sec

                if span_dur < profile.min_clip_sec * 0.8:
                    continue
                if span_dur > profile.max_clip_sec * 1.1:
                    break

                span_text = " ".join(c.text for c in window_cues[i : j + 1])
                score, hook_s, dens_s, comp_s, reason = self._score_span_heuristics(
                    window_cues[i : j + 1], span_text, span_dur, target_min, target_max
                )

                if score >= 0.55:
                    title = self._generate_heuristic_title(window_cues[i].text, span_text)
                    candidates.append(
                        ScoredSegment(
                            start_row=base_offset + i,
                            end_row=base_offset + j,
                            title=title,
                            reason=reason,
                            score=score,
                            hook_score=hook_s,
                            density_score=dens_s,
                            completeness_score=comp_s,
                            tags=["#AutoClip", "#ViralMoment", "#Highlights"],
                        )
                    )

        # Sort by score descending and take top 5 per window
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:5]

    def _score_span_heuristics(
        self,
        cues: List[TranscriptCue],
        span_text: str,
        duration_sec: float,
        target_min: float,
        target_max: float,
    ) -> Tuple[float, float, float, float, str]:
        """Calculates viral scores across hook, density, emotion, and structural completeness."""
        # 1. Hook Score (0.0 to 1.0)
        first_cue_text = cues[0].text.lower()
        hook_score = 0.5
        reasons = []

        for pat in HOOK_PATTERNS:
            if re.search(pat, first_cue_text):
                hook_score += 0.25
                reasons.append("Strong hook opening pattern")
                break

        if "?" in first_cue_text or "？" in first_cue_text or QUESTION_STARTERS.match(first_cue_text):
            hook_score += 0.2
            reasons.append("Intriguing opening question")

        hook_score = min(1.0, hook_score)

        # 2. Information Density Score (0.0 to 1.0)
        words = re.findall(r"\w+", span_text.lower())
        word_count = len(words)
        unique_words = len(set(words))
        diversity = (unique_words / max(1, word_count))

        # Check for numbers/metrics
        numbers_found = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", span_text))
        density_score = min(1.0, 0.4 + (diversity * 0.4) + min(0.2, numbers_found * 0.05))
        if numbers_found > 0:
            reasons.append("Contains concrete data/metrics")

        # 3. Emotional Tone Score (0.0 to 1.0)
        emotion_hits = sum(1 for w in words if w in EMOTION_KEYWORDS)
        exclamations = span_text.count("!") + span_text.count("！")
        emotion_score = min(1.0, 0.4 + (emotion_hits * 0.1) + min(0.2, exclamations * 0.05))

        # 4. Standalone Completeness Score (0.0 to 1.0)
        # Prefers starting with a capital letter/number and ending with sentence terminus (. ! ?)
        last_char = cues[-1].text.strip()[-1:] if cues[-1].text.strip() else ""
        has_proper_ending = last_char in {".", "!", "?", "。", "！", "？"}
        completeness_score = 0.9 if has_proper_ending else 0.65

        # 5. Duration Proximity Penalty
        duration_penalty = 0.0
        if duration_sec < target_min:
            duration_penalty = (target_min - duration_sec) / target_min * 0.15
        elif duration_sec > target_max:
            duration_penalty = (duration_sec - target_max) / target_max * 0.15

        # Aggregate weighted score
        final_score = (
            (hook_score * 0.35)
            + (density_score * 0.25)
            + (emotion_score * 0.20)
            + (completeness_score * 0.20)
            - duration_penalty
        )
        final_score = max(0.1, min(1.0, final_score))

        justification = "; ".join(reasons) if reasons else "High information density and balanced pacing"
        return final_score, hook_score, density_score, completeness_score, justification

    def _generate_heuristic_title(self, opening_cue: str, full_snippet: str) -> str:
        """Derives a punchy social title from the highlight content."""
        clean_opening = re.sub(r"^[0-9\.\-\:\s]+", "", opening_cue).strip()
        words = clean_opening.split()

        # If opening is already a punchy phrase, use it
        if 3 <= len(words) <= 10:
            title = " ".join(words)
        elif len(words) > 10:
            title = " ".join(words[:8]) + "..."
        else:
            # Fall back to snippet keywords
            first_sentence = re.split(r"[.!?。！？]", full_snippet)[0].strip()
            title = first_sentence[:45] if first_sentence else "Must Watch Moment"

        # Capitalize and strip formatting
        title = re.sub(r'[*_`#"]+', '', title).strip()
        return title[:60]

    def _evaluate_with_llm(
        self,
        window_cues: List[TranscriptCue],
        profile: DurationProfile,
        base_offset: int,
    ) -> List[ScoredSegment]:
        """Calls LLM with compact cues and parses JSON response."""
        rows_text = TranscriptSlicer.format_rows(window_cues)
        prompt = self.AUTOCLIP_PROMPT_TEMPLATE.format(
            min_sec=int(profile.min_clip_sec),
            max_sec=int(profile.max_clip_sec),
            rows=rows_text,
        )

        response_text = self.llm_callable(prompt) if self.llm_callable else ""
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            return []

        data = json.loads(json_match.group(0))
        raw_clips = data.get("clips", [])
        results: List[ScoredSegment] = []

        for item in raw_clips:
            s_idx = item.get("start")
            e_idx = item.get("end")
            if s_idx is None or e_idx is None or not (0 <= s_idx <= e_idx < len(window_cues)):
                continue

            score = float(item.get("score", 0.7))
            results.append(
                ScoredSegment(
                    start_row=base_offset + s_idx,
                    end_row=base_offset + e_idx,
                    title=str(item.get("title", f"Clip #{len(results)+1}")),
                    reason=str(item.get("reason", "Selected by AI model")),
                    score=max(0.0, min(1.0, score)),
                    hook_score=score,
                    density_score=score,
                    completeness_score=0.9,
                    tags=["#AIClip", "#Highlights"],
                )
            )

        return results

    def _extract_default_span(self, cues: List[TranscriptCue], profile: DurationProfile) -> Dict[str, Any]:
        """Constructs a default fallback highlight clip covering the most balanced window."""
        target_dur = profile.target_clip_sec[0]
        start_idx = 0
        end_idx = min(len(cues) - 1, 10)

        for i, c in enumerate(cues):
            if c.end_sec - cues[0].start_sec >= target_dur:
                end_idx = i
                break

        return {
            "start_row": start_idx,
            "end_row": end_idx,
            "title": cues[0].text[:40] if cues else "Key Video Highlight",
            "reason": "Primary introductory segment",
            "score": 0.70,
            "tags": ["#Highlight", "#KeyMoment"],
        }
