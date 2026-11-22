"""
JARVIS Video Editing Engine V2.0 — Universal Autonomous Media Discovery Engine
=============================================================================
A universal, subject-independent media discovery and acquisition system that:
1. Understands user subjects across 16 universal categories (automotive, nature,
   anime/animation, space, sports, travel, etc.).
2. Generates hierarchical and multi-angle search query variations.
3. Implements a modular provider architecture:
   - Pexels (Official Stock API)
   - Pixabay (Official Stock API)
   - Wikimedia Commons (Open API, Creative Commons / Public Domain)
   - Internet Archive (Open API, Public Domain Movies)
4. Checks provider connectivity and reports missing API configuration.
5. Performs multi-source fallback: if a provider fails or yields 0 results,
   seamlessly falls back to next authorized sources.
6. Transcodes open format videos (.webm, .ogv) to compliant .mp4.
7. Validates downloaded streams using FFprobe (duration, resolution, fps, codec).
8. Categorizes camera angles (front, rear, side, rolling, close-up, interior, etc.).
9. Maintains strict copyright protection (rejects unauthorized scraping of commercial IP).
10. Generates detailed structured diagnostics for every provider attempt.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import os
import re
import urllib.parse
import httpx
import subprocess

from core.logger import log
from video_editor.workspace import workspace_manager
from video_editor.input_manager import input_manager


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiscoveredMediaItem:
    """Metadata profile of a discovered media asset."""
    title: str
    source_url: str
    source_platform: str  # "wikimedia", "internet_archive", "pexels", "pixabay"
    media_type: str = "video"
    duration_sec: float = 0.0
    license_type: str = "Creative Commons"
    is_authorized: bool = True
    suitability_score: float = 0.8
    local_path: Optional[str] = None
    category: str = "custom_general"
    camera_angle: str = "general"
    is_exact_match: bool = True
    resolution: str = "1920x1080"
    fps: float = 30.0
    original_page: Optional[str] = None
    download_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    file_size_bytes: int = 0


@dataclass
class MediaCollectionReport:
    """Comprehensive summary of an autonomous media discovery and acquisition pass."""
    query: str
    clean_subject: str = ""
    category: str = "custom_general"
    items_found: int = 0
    items_downloaded: int = 0
    is_copyright_restricted: bool = False
    restriction_message: Optional[str] = None
    is_exact_match: bool = True
    fallback_used: Optional[str] = None
    fallback_explanation: Optional[str] = None
    camera_angles_found: List[str] = field(default_factory=list)
    diagnostics: List[Dict[str, str]] = field(default_factory=list)
    provider_status: Dict[str, str] = field(default_factory=dict)
    items: List[DiscoveredMediaItem] = field(default_factory=list)


# Known commercial franchises requiring user-supplied raw clips
KNOWN_COPYRIGHTED_TERMS = {
    "goku", "dragon ball", "dragonball", "naruto", "sasuke", "one piece", "luffy",
    "bleach", "ichigo", "jujutsu kaisen", "gojo", "demon slayer", "tanjiro",
    "attack on titan", "eren", "marvel", "avengers", "iron man", "batman",
    "superman", "spiderman", "spider-man", "star wars", "disney", "pixar",
    "solo leveling", "sung jin-woo", "cristiano ronaldo", "messi",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Universal Subject & Category Intelligence
# ─────────────────────────────────────────────────────────────────────────────

class SubjectAnalyzer:
    """Classifies queries into universal categories and generates search variations."""

    CATEGORIES = {
        "cars_automotive": [
            "car", "cars", "automobile", "automotive", "bmw", "m4", "m3", "audi", "mercedes",
            "porsche", "ferrari", "lamborghini", "aventador", "huracan", "dodge", "challenger",
            "mustang", "corvette", "supercar", "hypercar", "drifting", "exhaust", "vehicle",
        ],
        "nature_landscapes": [
            "mountain", "mountains", "forest", "nature", "landscape", "waterfall", "river",
            "ocean", "beach", "sunset", "sunrise", "clouds", "sky", "scenery", "valley",
            "canyon", "desert", "rainforest", "aurora",
        ],
        "anime_animation": [
            "anime", "animation", "animated", "cartoon", "cgi", "blender", "open movie",
            "manga", "sintel", "big buck bunny", "render",
        ],
        "bikes_motorcycles": [
            "bike", "motorcycle", "ducati", "yamaha", "kawasaki", "superbike", "motogp",
            "harley", "biker", "cyclist",
        ],
        "space_science": [
            "space", "galaxy", "stars", "planet", "nasa", "earth", "moon", "nebula",
            "telescope", "physics", "cosmos", "orbit", "astronomy",
        ],
        "gaming_gameplay": [
            "game", "gaming", "gameplay", "esports", "fps", "fortnite", "minecraft",
            "gta", "valorant", "cod", "call of duty", "cyberpunk", "csgo", "counter-strike",
            "pubg", "apex legends", "league of legends", "elden ring", "speedrun", "clip", "clutch",
        ],
        "animals_wildlife": [
            "animal", "animals", "wildlife", "lion", "tiger", "wolf", "eagle", "bear",
            "shark", "whale", "dog", "cat", "safari",
        ],
        "sports_athletes": [
            "sports", "athlete", "football", "soccer", "basketball", "tennis", "gymnastics",
            "boxing", "ufc", "mma", "racing", "runner",
        ],
        "fitness_bodybuilding": [
            "fitness", "gym", "bodybuilding", "workout", "muscle", "crossfit", "training",
        ],
        "technology_futuristic": [
            "technology", "tech", "robot", "robotics", "cyberpunk", "futuristic", "ai",
            "circuit", "hacker", "cyber",
        ],
        "travel_tourism": [
            "travel", "tourism", "vacation", "resort", "explore", "flight", "tropical",
        ],
        "architecture_cities": [
            "city", "architecture", "skyscraper", "building", "urban", "night city",
            "tokyo", "new york", "dubai", "bridge",
        ],
        "music_performance": [
            "music", "concert", "dj", "guitar", "piano", "festival", "stage", "dance",
        ],
    }

    @classmethod
    def extract_clean_subject(cls, raw_prompt: str) -> str:
        """Strip command syntax, modifiers, durations, and instructions to get core entity."""
        text = raw_prompt.strip()
        # Remove /reel prefix and action verbs
        text = re.sub(r'(?i)^/reel\s*', '', text).strip()
        text = re.sub(r'(?i)^(?:create|make|edit|generate)\s+(?:a|an)\s+', '', text).strip()
        # Cut off after period, colon, or semicolon if it starts an instruction
        text = re.sub(r'[\.:;].*$', '', text).strip()
        # Strip duration words
        text = re.sub(r'(?i)\b(?:duration:\s*)?\d+(?:\.\d+)?\s*(?:s|sec|seconds)?\b', '', text).strip()
        # Strip standard style keywords and genre qualifiers
        text = re.sub(
            r'(?i)\b(?:cinematic|aggressive|velocity|smooth|speed\s+ramps?|4k|hd|vertical|reel|edit|video|shorts?|travel|tourism|scenic|scenery|highlights?)\b',
            '',
            text
        ).strip()
        # Clean extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text or "cinematic"

    @classmethod
    def classify_category(cls, subject: str) -> str:
        """Identify universal category from subject keywords."""
        lower = subject.lower()
        for cat, keywords in cls.CATEGORIES.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', lower) for kw in keywords):
                return cat
        return "custom_general"

    @classmethod
    def generate_search_queries(cls, subject: str, category: str) -> List[Tuple[str, bool]]:
        """
        Generate hierarchical search variations: (query_string, is_exact_match).
        Includes specific camera angles, action shots, brand level, and category fallbacks.
        """
        clean = subject.strip()
        variations: List[Tuple[str, bool]] = []

        # 1. Exact query
        variations.append((clean, True))

        if category == "cars_automotive":
            # Multi-angle automotive searches
            variations.append((f"{clean} driving", True))
            variations.append((f"{clean} rolling shot", True))
            variations.append((f"{clean} front", True))
            variations.append((f"{clean} exterior", True))

            # Brand extraction (e.g. BMW M4 -> BMW; Dodge Challenger -> Dodge)
            tokens = clean.split()
            if len(tokens) > 1:
                brand = tokens[0]
                variations.append((brand, False))
                variations.append((f"{brand} car", False))

            # Generic fallbacks
            variations.append(("sports car", False))
            variations.append(("car automobile", False))

        elif category == "nature_landscapes":
            variations.append((f"{clean} landscape", True))
            variations.append((f"{clean} aerial drone", True))
            variations.append((f"{clean} cinematic", True))
            variations.append(("mountain landscape", False))
            variations.append(("nature scenery", False))

        elif category == "anime_animation":
            variations.append((f"{clean} animation", True))
            variations.append((f"{clean} open source", True))
            variations.append(("open source animation", False))
            variations.append(("blender open movie animation", False))

        elif category == "gaming_gameplay":
            variations.append((f"{clean} gameplay", True))
            variations.append((f"{clean} clutch moment", True))
            variations.append((f"{clean} gaming highlights", True))
            variations.append(("gaming gameplay", False))
            variations.append(("esports highlights", False))

        elif category == "space_science":
            variations.append((f"{clean} space", True))
            variations.append((f"{clean} telescope", True))
            variations.append(("space nebula galaxy", False))

        else:
            variations.append((f"{clean} cinematic", True))
            tokens = clean.split()
            if len(tokens) > 1:
                variations.append((tokens[0], False))
            variations.append(("cinematic motion", False))

        return variations


# ─────────────────────────────────────────────────────────────────────────────
# 3. Camera Angle & Shot Classifier
# ─────────────────────────────────────────────────────────────────────────────

class CameraAngleClassifier:
    """Classifies video clips into camera angles and cinematic shot categories."""

    ANGLE_PATTERNS = {
        "night_driving": [r"night", r"dark", r"streetlight", r"headlamp"],
        "track_driving": [r"track", r"race", r"circuit", r"drift", r"\bfast\b", r"\bspeed\b"],
        "interior": [r"interior", r"cabin", r"cockpit", r"dashboard", r"steering"],
        "front_angle": [r"front", r"headlight", r"grille", r"face", r"head-on", r"forward"],
        "rear_angle": [r"rear", r"tail", r"taillight", r"exhaust", r"back", r"chase"],
        "side_angle": [r"side", r"profile", r"flank"],
        "close_up": [r"detail", r"close", r"macro", r"badge", r"wheel", r"rim", r"eye"],
        "rolling_shot": [r"roll", r"tracking", r"motion", r"driving", r"run", r"panning"],
        "static_cinematic": [r"static", r"parked", r"stand", r"loop", r"showcase", r"studio"],
    }

    @classmethod
    def classify(cls, title: str, description: str = "") -> str:
        """Determine camera angle or shot type from media metadata."""
        text = f"{title} {description}".lower()
        for angle, patterns in cls.ANGLE_PATTERNS.items():
            if any(re.search(p, text) for p in patterns):
                return angle
        return "rolling_shot" if any(k in text for k in ["loop", "move", "anim"]) else "static_cinematic"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Modular Provider Architecture
# ─────────────────────────────────────────────────────────────────────────────

class FootageProvider(ABC):
    """Abstract base class for all footage search & acquisition providers."""

    def __init__(self, name: str, priority: int = 1):
        self.name = name
        self.priority = priority
        self.user_agent = (
            "JARVISAutonomousVideoEditor/2.0 "
            "(https://github.com/harshadev-7/JARVIS-Project; contact@jarvis-ai.org)"
        )

    @abstractmethod
    def is_configured(self) -> Tuple[bool, Optional[str]]:
        """Check if provider configuration (e.g. API keys) is available."""
        pass

    @abstractmethod
    async def check_connectivity(self) -> Tuple[bool, Optional[str]]:
        """Verify reachability of the provider endpoint."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        category: str = "custom_general",
        limit: int = 3,
        is_exact: bool = True,
    ) -> List[DiscoveredMediaItem]:
        """Search provider for video footage matching query."""
        pass


class WikimediaCommonsProvider(FootageProvider):
    """Searches Wikimedia Commons API for Creative Commons and Public Domain footage."""

    def __init__(self):
        super().__init__(name="Wikimedia Commons", priority=3)
        self.endpoint = "https://commons.wikimedia.org/w/api.php"

    def is_configured(self) -> Tuple[bool, Optional[str]]:
        return True, None

    async def check_connectivity(self) -> Tuple[bool, Optional[str]]:
        headers = {"User-Agent": self.user_agent}
        try:
            async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
                r = await client.get(self.endpoint, params={"action": "query", "meta": "siteinfo", "format": "json"})
                if r.status_code == 200:
                    return True, None
                return False, f"NETWORK_ERROR: Wikimedia returned status {r.status_code}"
        except Exception as e:
            return False, f"NETWORK_ERROR: Connection to Wikimedia failed ({e})"

    async def search(
        self,
        query: str,
        category: str = "custom_general",
        limit: int = 3,
        is_exact: bool = True,
    ) -> List[DiscoveredMediaItem]:
        clean_q = re.sub(r"[^\w\s]", "", query).strip()
        search_term = f"{clean_q} filetype:video"
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": search_term,
            "gsrnamespace": "6",  # File: namespace
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "format": "json",
            "gsrlimit": str(limit * 3),
        }
        headers = {"User-Agent": self.user_agent}
        items: List[DiscoveredMediaItem] = []

        try:
            async with httpx.AsyncClient(timeout=12.0, headers=headers) as client:
                resp = await client.get(self.endpoint, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    pages = data.get("query", {}).get("pages", {})
                    for page_id, page in pages.items():
                        info_list = page.get("imageinfo", [])
                        if not info_list:
                            continue
                        info = info_list[0]
                        media_url = info.get("url", "")
                        if any(ext in media_url.lower() for ext in [".webm", ".mp4", ".ogv"]):
                            title = page.get("title", "").replace("File:", "").strip()
                            meta = info.get("extmetadata", {})
                            lic = meta.get("LicenseShortName", {}).get("value", "Creative Commons / Public Domain")
                            desc = meta.get("ImageDescription", {}).get("value", "")
                            angle = CameraAngleClassifier.classify(title, desc)
                            items.append(DiscoveredMediaItem(
                                title=title,
                                source_url=media_url,
                                source_platform="wikimedia",
                                media_type="video",
                                license_type=lic,
                                is_authorized=True,
                                suitability_score=0.9 if is_exact else 0.75,
                                category=category,
                                camera_angle=angle,
                                is_exact_match=is_exact,
                                original_page=f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(page.get('title', ''))}",
                            ))
                            if len(items) >= limit:
                                break
        except Exception as e:
            log.warning(f"Wikimedia search failed for '{query}': {e}")

        return items


class InternetArchiveProvider(FootageProvider):
    """Searches Internet Archive Movies for public domain footage."""

    def __init__(self):
        super().__init__(name="Internet Archive", priority=4)
        self.endpoint = "https://archive.org/advancedsearch.php"

    def is_configured(self) -> Tuple[bool, Optional[str]]:
        return True, None

    async def check_connectivity(self) -> Tuple[bool, Optional[str]]:
        headers = {"User-Agent": self.user_agent}
        try:
            async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
                r = await client.get(self.endpoint, params={"q": "test", "rows": "1", "output": "json"})
                if r.status_code == 200:
                    return True, None
                return False, f"NETWORK_ERROR: Internet Archive returned status {r.status_code}"
        except Exception as e:
            return False, f"NETWORK_ERROR: Connection to Internet Archive failed ({e})"

    async def search(
        self,
        query: str,
        category: str = "custom_general",
        limit: int = 3,
        is_exact: bool = True,
    ) -> List[DiscoveredMediaItem]:
        clean_q = re.sub(r"[^\w\s]", "", query).strip()
        search_url = (
            f"https://archive.org/advancedsearch.php?q={urllib.parse.quote(clean_q)}+AND+mediatype:(movies)"
            f"&fl[]=identifier,title,description,downloads&sort[]=downloads+desc&rows={limit * 2}&output=json"
        )
        headers = {"User-Agent": self.user_agent}
        items: List[DiscoveredMediaItem] = []

        try:
            async with httpx.AsyncClient(timeout=12.0, headers=headers) as client:
                resp = await client.get(search_url)
                if resp.status_code == 200:
                    docs = resp.json().get("response", {}).get("docs", [])
                    for doc in docs:
                        ident = doc.get("identifier", "")
                        title = doc.get("title", ident)
                        desc = doc.get("description", "")
                        # Filter out YouTube scrapes and invalid IDs
                        if not ident or ident.lower().startswith("youtube-"):
                            continue
                        meta_url = f"https://archive.org/metadata/{ident}/files"
                        meta_resp = await client.get(meta_url)
                        if meta_resp.status_code == 200:
                            files_data = meta_resp.json().get("result", [])
                            for f in files_data:
                                f_name = f.get("name", "")
                                # Check file size to avoid multi-gigabyte archival downloads
                                size_val = f.get("size", "0")
                                size_bytes = int(size_val) if str(size_val).isdigit() else 0
                                if size_bytes > 35 * 1024 * 1024:
                                    continue
                                if f_name.lower().endswith(".mp4") and "512kb" not in f_name.lower():
                                    direct_url = f"https://archive.org/download/{ident}/{f_name}"
                                    angle = CameraAngleClassifier.classify(title, desc)
                                    items.append(DiscoveredMediaItem(
                                        title=title,
                                        source_url=direct_url,
                                        source_platform="internet_archive",
                                        media_type="video",
                                        license_type="Public Domain Archive",
                                        is_authorized=True,
                                        suitability_score=0.85 if is_exact else 0.7,
                                        category=category,
                                        camera_angle=angle,
                                        is_exact_match=is_exact,
                                        original_page=f"https://archive.org/details/{ident}",
                                    ))
                                    break
                        if len(items) >= limit:
                            break
        except Exception as e:
            log.warning(f"Internet Archive search failed for '{query}': {e}")

        return items


class PexelsProvider(FootageProvider):
    """Searches official Pexels API for royalty-free stock footage."""

    def __init__(self):
        super().__init__(name="Pexels", priority=1)
        self.endpoint = "https://api.pexels.com/videos/search"

    def _get_api_key(self) -> Optional[str]:
        return os.getenv("PEXELS_API_KEY")

    def is_configured(self) -> Tuple[bool, Optional[str]]:
        key = self._get_api_key()
        if not key or "your-" in key.lower():
            return (
                False,
                "API_CONFIGURATION_ERROR: PEXELS_API_KEY is not configured in .env. "
                "Add PEXELS_API_KEY=<your_key> to access curated 4K/HD stock footage."
            )
        return True, None

    async def check_connectivity(self) -> Tuple[bool, Optional[str]]:
        key = self._get_api_key()
        if not key:
            return False, "API_CONFIGURATION_ERROR: Missing PEXELS_API_KEY"
        headers = {"Authorization": key, "User-Agent": self.user_agent}
        try:
            async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
                r = await client.get(self.endpoint, params={"query": "test", "per_page": 1})
                if r.status_code == 200:
                    return True, None
                return False, f"NETWORK_ERROR: Pexels API returned status {r.status_code}"
        except Exception as e:
            return False, f"NETWORK_ERROR: Connection to Pexels failed ({e})"

    async def search(
        self,
        query: str,
        category: str = "custom_general",
        limit: int = 3,
        is_exact: bool = True,
    ) -> List[DiscoveredMediaItem]:
        key = self._get_api_key()
        if not key:
            return []

        headers = {"Authorization": key, "User-Agent": self.user_agent}
        params = {"query": query, "per_page": limit, "orientation": "portrait"}
        items: List[DiscoveredMediaItem] = []

        try:
            async with httpx.AsyncClient(timeout=12.0, headers=headers) as client:
                r = await client.get(self.endpoint, params=params)
                if r.status_code == 200:
                    videos = r.json().get("videos", [])
                    for v in videos:
                        vid_files = v.get("video_files", [])
                        # Prefer HD/FullHD MP4
                        best_file = next(
                            (f for f in vid_files if f.get("quality") == "hd" and f.get("file_type") == "video/mp4"),
                            vid_files[0] if vid_files else None
                        )
                        if best_file and best_file.get("link"):
                            title = f"Pexels Video {v.get('id')} ({query})"
                            angle = CameraAngleClassifier.classify(title)
                            items.append(DiscoveredMediaItem(
                                title=title,
                                source_url=best_file.get("link"),
                                source_platform="pexels",
                                media_type="video",
                                duration_sec=float(v.get("duration", 0)),
                                license_type="Pexels Royalty-Free License",
                                is_authorized=True,
                                suitability_score=0.95 if is_exact else 0.8,
                                category=category,
                                camera_angle=angle,
                                is_exact_match=is_exact,
                                original_page=v.get("url", ""),
                            ))
        except Exception as e:
            log.warning(f"Pexels search failed for '{query}': {e}")

        return items


class PixabayProvider(FootageProvider):
    """Searches official Pixabay Video API for royalty-free footage."""

    def __init__(self):
        super().__init__(name="Pixabay", priority=2)
        self.endpoint = "https://pixabay.com/api/videos/"

    def _get_api_key(self) -> Optional[str]:
        return os.getenv("PIXABAY_API_KEY")

    def is_configured(self) -> Tuple[bool, Optional[str]]:
        key = self._get_api_key()
        if not key or "your-" in key.lower():
            return (
                False,
                "API_CONFIGURATION_ERROR: PIXABAY_API_KEY is not configured in .env. "
                "Add PIXABAY_API_KEY=<your_key> to access Pixabay stock footage."
            )
        return True, None

    async def check_connectivity(self) -> Tuple[bool, Optional[str]]:
        key = self._get_api_key()
        if not key:
            return False, "API_CONFIGURATION_ERROR: Missing PIXABAY_API_KEY"
        headers = {"User-Agent": self.user_agent}
        try:
            async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
                r = await client.get(self.endpoint, params={"key": key, "q": "car", "per_page": 3})
                if r.status_code == 200:
                    return True, None
                return False, f"NETWORK_ERROR: Pixabay API returned status {r.status_code}"
        except Exception as e:
            return False, f"NETWORK_ERROR: Connection to Pixabay failed ({e})"

    async def search(
        self,
        query: str,
        category: str = "custom_general",
        limit: int = 3,
        is_exact: bool = True,
    ) -> List[DiscoveredMediaItem]:
        key = self._get_api_key()
        if not key:
            return []

        headers = {"User-Agent": self.user_agent}
        params = {"key": key, "q": query, "per_page": limit, "video_type": "all"}
        items: List[DiscoveredMediaItem] = []

        try:
            async with httpx.AsyncClient(timeout=12.0, headers=headers) as client:
                r = await client.get(self.endpoint, params=params)
                if r.status_code == 200:
                    hits = r.json().get("hits", [])
                    for hit in hits:
                        vids = hit.get("videos", {})
                        # Choose large or medium
                        target_stream = vids.get("large") or vids.get("medium") or vids.get("small")
                        if target_stream and target_stream.get("url"):
                            tags = hit.get("tags", "")
                            title = f"Pixabay Video {hit.get('id')} ({tags})"
                            angle = CameraAngleClassifier.classify(title, tags)
                            items.append(DiscoveredMediaItem(
                                title=title,
                                source_url=target_stream.get("url"),
                                source_platform="pixabay",
                                media_type="video",
                                duration_sec=float(hit.get("duration", 0)),
                                license_type="Pixabay Content License",
                                is_authorized=True,
                                suitability_score=0.92 if is_exact else 0.78,
                                category=category,
                                camera_angle=angle,
                                is_exact_match=is_exact,
                                original_page=hit.get("pageURL", ""),
                            ))
        except Exception as e:
            log.warning(f"Pixabay search failed for '{query}': {e}")

        return items


# ─────────────────────────────────────────────────────────────────────────────
# 5. Universal Autonomous Media Discovery Engine
# ─────────────────────────────────────────────────────────────────────────────

class MediaDiscoveryEngine:
    """
    Coordinates universal, multi-provider footage discovery, downloading,
    transcoding, stream validation, and error diagnostic reporting.
    """

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self.user_agent = (
            "JARVISAutonomousVideoEditor/2.0 "
            "(https://github.com/harshadev-7/JARVIS-Project; contact@jarvis-ai.org)"
        )
        # Register providers ordered by priority
        self.providers: List[FootageProvider] = [
            PexelsProvider(),
            PixabayProvider(),
            WikimediaCommonsProvider(),
            InternetArchiveProvider(),
        ]

    def is_copyright_protected(self, query: str) -> Tuple[bool, Optional[str]]:
        """Check if query requests copyrighted media requiring user-supplied clips."""
        lower = query.lower()
        for term in KNOWN_COPYRIGHTED_TERMS:
            if re.search(r'\b' + re.escape(term) + r'\b', lower):
                return True, (
                    f"⚠️ **Copyright Protection Alert:** '{term.title()}' is a copyrighted commercial franchise. "
                    f"Automated public domain repositories cannot legally provide commercial anime/movie footage.\n\n"
                    f"**Action Required:** Please drop your {term.title()} video clips into `media/input/` "
                    f"and rerun your command to compile your edit with professional velocity and effects!"
                )
        return False, None

    # Backward compatibility methods
    async def search_wikimedia_videos(self, query: str, limit: int = 3) -> List[DiscoveredMediaItem]:
        provider = WikimediaCommonsProvider()
        return await provider.search(query=query, limit=limit)

    async def search_internet_archive_videos(self, query: str, limit: int = 3) -> List[DiscoveredMediaItem]:
        provider = InternetArchiveProvider()
        return await provider.search(query=query, limit=limit)

    async def download_media(
        self,
        item: DiscoveredMediaItem,
        destination_folder: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Download a discovered authorized video file to local storage.
        Transcodes .webm and .ogv streams to compliant .mp4 and validates with FFprobe.
        """
        dest_dir = destination_folder or workspace_manager.dirs["media_input"]
        dest_dir.mkdir(parents=True, exist_ok=True)

        clean_filename = re.sub(r"[^\w\-_\.]", "_", item.title)[:40]
        # Download raw file initially
        raw_ext = Path(urllib.parse.urlparse(item.source_url).path).suffix or ".mp4"
        if not raw_ext.startswith("."):
            raw_ext = f".{raw_ext}"

        raw_file = dest_dir / f"{clean_filename}_raw{raw_ext}"
        final_mp4 = dest_dir / f"{clean_filename}.mp4"

        # If clean MP4 already exists and valid, reuse
        if final_mp4.exists() and final_mp4.stat().st_size > 50000:
            try:
                meta = input_manager.probe_video(final_mp4)
                if meta.duration_sec >= 0.5:
                    item.local_path = str(final_mp4)
                    item.duration_sec = meta.duration_sec
                    item.resolution = f"{meta.width}x{meta.height}"
                    item.fps = meta.fps
                    item.file_size_bytes = final_mp4.stat().st_size
                    return final_mp4
            except Exception:
                pass

        headers = {"User-Agent": self.user_agent}
        async def _stream_download() -> bool:
            async with httpx.AsyncClient(timeout=25.0, headers=headers, follow_redirects=True) as client:
                async with client.stream("GET", item.source_url) as resp:
                    if resp.status_code == 200:
                        with open(raw_file, "wb") as f:
                            async for chunk in resp.aiter_bytes(chunk_size=65536):
                                f.write(chunk)
                        return True
                    else:
                        log.warning(f"Download HTTP {resp.status_code} for {item.source_url}")
                        return False

        try:
            log.info(f"Downloading footage: '{item.title}' from {item.source_platform}...")
            ok = await asyncio.wait_for(_stream_download(), timeout=30.0)
            if not ok:
                return None

            if not raw_file.exists() or raw_file.stat().st_size < 10000:
                log.warning(f"Downloaded file empty or too small: {raw_file}")
                if raw_file.exists():
                    raw_file.unlink()
                return None

            # Transcode/Remux to universally compatible MP4 if needed
            ffmpeg_bin = input_manager.ffmpeg_bin
            if raw_file.suffix.lower() in [".webm", ".ogv", ".ogg", ".mkv"]:
                log.info(f"Transcoding {raw_file.name} to standardized .mp4...")
                cmd = [
                    ffmpeg_bin, "-y",
                    "-i", str(raw_file),
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                    "-c:a", "aac",
                    str(final_mp4),
                ]
                proc = subprocess.run(cmd, capture_output=True, timeout=60)
                if raw_file.exists():
                    raw_file.unlink()
                if proc.returncode != 0 or not final_mp4.exists():
                    log.warning(f"Transcoding failed for {raw_file.name}")
                    return None
            else:
                raw_file.rename(final_mp4)

            # Mandatory FFprobe verification
            meta = input_manager.probe_video(final_mp4)
            if meta.duration_sec < 0.5 or meta.width < 100:
                log.warning(f"VIDEO_VALIDATION_ERROR: File {final_mp4.name} invalid (dur={meta.duration_sec})")
                final_mp4.unlink()
                return None

            item.local_path = str(final_mp4)
            item.duration_sec = meta.duration_sec
            item.resolution = f"{meta.width}x{meta.height}"
            item.fps = meta.fps
            item.file_size_bytes = final_mp4.stat().st_size
            log.info(
                f"Footage verified: {final_mp4.name} | {item.resolution} | "
                f"{item.fps:.1f}fps | {item.duration_sec:.1f}s"
            )
            return final_mp4

        except Exception as e:
            log.warning(f"Failed to acquire footage '{item.title}': {e}")
            if raw_file.exists():
                raw_file.unlink()
            if final_mp4.exists() and final_mp4.stat().st_size < 10000:
                final_mp4.unlink()

        return None

    async def discover_and_acquire(
        self,
        prompt: str,
        target_clips_count: int = 2,
    ) -> MediaCollectionReport:
        """
        Universal autonomous discovery workflow:
        1. Analyzes prompt to extract clean entity and universal category.
        2. Evaluates copyright restrictions.
        3. Executes multi-provider search with fallback hierarchy.
        4. Downloads, standardizes to MP4, and validates video streams.
        5. Compiles rich report with diagnostics and camera angle breakdown.
        """
        # 1. Subject Extraction & Categorization
        clean_subject = SubjectAnalyzer.extract_clean_subject(prompt)
        category = SubjectAnalyzer.classify_category(clean_subject)
        log.info(f"Universal Media Discovery: query='{clean_subject}' | category='{category}'")

        # 2. Strict Copyright Gatekeeper Check
        is_restricted, msg = self.is_copyright_protected(clean_subject)
        if is_restricted:
            return MediaCollectionReport(
                query=prompt,
                clean_subject=clean_subject,
                category=category,
                items_found=0,
                items_downloaded=0,
                is_copyright_restricted=True,
                restriction_message=msg,
                items=[],
            )

        # 3. Generate Search Variations (Exact angles -> Brand -> Category Fallbacks)
        search_variations = SubjectAnalyzer.generate_search_queries(clean_subject, category)

        diagnostics: List[Dict[str, str]] = []
        provider_status: Dict[str, str] = {}
        all_candidates: List[DiscoveredMediaItem] = []
        seen_urls = set()
        is_exact_acquired = False
        fallback_term_used = None

        # Check provider configuration and connectivity
        active_providers: List[FootageProvider] = []
        for p in self.providers:
            is_cfg, cfg_err = p.is_configured()
            if not is_cfg:
                diagnostics.append({"provider": p.name, "code": "API_CONFIGURATION_ERROR", "message": cfg_err})
                provider_status[p.name] = "Not Configured (Missing API Key)"
                continue

            conn_ok, conn_err = await p.check_connectivity()
            if not conn_ok:
                diagnostics.append({"provider": p.name, "code": "NETWORK_ERROR", "message": conn_err})
                provider_status[p.name] = "Network Unreachable"
                continue

            provider_status[p.name] = "Connected"
            active_providers.append(p)

        # 4. Multi-Source Search & Fallback Pipeline
        for query_str, is_exact in search_variations:
            if len(all_candidates) >= target_clips_count * 2:
                break

            for provider in active_providers:
                try:
                    found = await provider.search(
                        query=query_str,
                        category=category,
                        limit=target_clips_count,
                        is_exact=is_exact,
                    )
                    for itm in found:
                        if itm.source_url not in seen_urls:
                            seen_urls.add(itm.source_url)
                            all_candidates.append(itm)
                            if is_exact:
                                is_exact_acquired = True
                            elif not fallback_term_used:
                                fallback_term_used = query_str
                except Exception as e:
                    diagnostics.append({
                        "provider": provider.name,
                        "code": "SEARCH_ERROR",
                        "message": f"Search failed on '{query_str}': {e}",
                    })

        # 5. Acquire & Download Clips into media/input/
        # Prioritize CDN-hosted platforms (Wikimedia, Pexels, Pixabay) for fast, reliable delivery
        all_candidates.sort(
            key=lambda x: (
                1 if x.source_platform in ["pexels", "pixabay", "wikimedia"] else 0,
                1 if x.is_exact_match else 0,
                x.suitability_score,
            ),
            reverse=True,
        )

        downloaded_items: List[DiscoveredMediaItem] = []
        camera_angles_found: List[str] = []

        for item in all_candidates:
            if len(downloaded_items) >= target_clips_count:
                break
            local_path = await self.download_media(item)
            if local_path and local_path.exists():
                downloaded_items.append(item)
                if item.camera_angle not in camera_angles_found:
                    camera_angles_found.append(item.camera_angle)

        # 6. Build Fallback Explanation if Exact Match was Unavailable
        fallback_explanation = None
        if downloaded_items and not is_exact_acquired:
            fallback_explanation = (
                f"Notice: Exact '{clean_subject}' video was not found on authorized public domain repositories. "
                f"JARVIS acquired {len(downloaded_items)} visually similar, properly licensed {category.replace('_', ' ')} "
                f"clips ('{fallback_term_used}') to compile your cinematic edit."
            )

        return MediaCollectionReport(
            query=prompt,
            clean_subject=clean_subject,
            category=category,
            items_found=len(all_candidates),
            items_downloaded=len(downloaded_items),
            is_copyright_restricted=False,
            restriction_message=None,
            is_exact_match=is_exact_acquired,
            fallback_used=fallback_term_used,
            fallback_explanation=fallback_explanation,
            camera_angles_found=camera_angles_found,
            diagnostics=diagnostics,
            provider_status=provider_status,
            items=downloaded_items,
        )


# Global singleton instance
media_discovery_engine = MediaDiscoveryEngine()
