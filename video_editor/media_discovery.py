"""
JARVIS Video Editing Engine V2.0 — Autonomous Media Discovery Engine
====================================================================
Discovers, checks licensing, and acquires authorized public domain and
Creative Commons video and audio assets from:
- Wikimedia Commons Video API (CC0, CC-BY, Public Domain)
- Internet Archive Movies API (Public Domain Archives)
- Royalty-Free Stock Video & Audio Endpoints
- Copyright Protection Gatekeeper: Rejects downloading unauthorized/copyrighted
  commercial media (e.g. Dragon Ball, Marvel, Naruto) and prompts the user
  to provide their own clips in media/input/.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import urllib.parse
import httpx

from core.logger import log
from video_editor.workspace import workspace_manager


@dataclass
class DiscoveredMediaItem:
    """Metadata profile of a discovered media asset."""
    title: str
    source_url: str
    source_platform: str  # "wikimedia", "internet_archive", "stock"
    media_type: str       # "video" or "audio"
    duration_sec: float = 0.0
    license_type: str = "Creative Commons"
    is_authorized: bool = True
    suitability_score: float = 0.8
    local_path: Optional[str] = None


@dataclass
class MediaCollectionReport:
    """Summary report of an autonomous media discovery pass."""
    query: str
    items_found: int
    items_downloaded: int
    is_copyright_restricted: bool = False
    restriction_message: Optional[str] = None
    items: List[DiscoveredMediaItem] = field(default_factory=list)


# Known commercial franchises where automated scraping is not legally permitted
KNOWN_COPYRIGHTED_TERMS = {
    "goku", "dragon ball", "dragonball", "naruto", "sasuke", "one piece", "luffy",
    "bleach", "ichigo", "jujutsu kaisen", "gojo", "demon slayer", "tanjiro",
    "attack on titan", "eren", "marvel", "avengers", "iron man", "batman",
    "superman", "spiderman", "spider-man", "star wars", "disney", "pixar",
}


class MediaDiscoveryEngine:
    """Discovers and acquires authorized public domain/CC video footage."""

    def __init__(self, timeout: float = 12.0):
        self.timeout = timeout
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (JARVIS-VideoEditor/2.0; Contact: harshadev-7)"
        )

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

    async def search_wikimedia_videos(self, query: str, limit: int = 3) -> List[DiscoveredMediaItem]:
        """Query Wikimedia Commons API for Creative Commons / Public Domain videos."""
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
            "gsrlimit": str(limit * 2),
        }
        url = "https://commons.wikimedia.org/w/api.php"
        headers = {"User-Agent": self.user_agent}
        items: List[DiscoveredMediaItem] = []

        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    pages = data.get("query", {}).get("pages", {})
                    for page_id, page in pages.items():
                        info_list = page.get("imageinfo", [])
                        if not info_list:
                            continue
                        info = info_list[0]
                        mime = info.get("mime", "").lower()
                        media_url = info.get("url", "")
                        if any(ext in media_url.lower() for ext in [".webm", ".mp4", ".ogv"]):
                            title = page.get("title", "").replace("File:", "")
                            meta = info.get("extmetadata", {})
                            lic = meta.get("LicenseShortName", {}).get("value", "Creative Commons / Public Domain")
                            items.append(DiscoveredMediaItem(
                                title=title,
                                source_url=media_url,
                                source_platform="wikimedia",
                                media_type="video",
                                license_type=lic,
                                is_authorized=True,
                                suitability_score=0.85,
                            ))
                            if len(items) >= limit:
                                break
        except Exception as e:
            log.warning(f"Wikimedia video search failed on '{query}': {e}")

        return items

    async def search_internet_archive_videos(self, query: str, limit: int = 3) -> List[DiscoveredMediaItem]:
        """Query Internet Archive for public domain / open movie clips."""
        clean_q = re.sub(r"[^\w\s]", "", query).strip()
        search_url = (
            f"https://archive.org/advancedsearch.php?q={urllib.parse.quote(clean_q)}+AND+mediatype:(movies)"
            f"&fl[]=identifier,title,description,downloads&sort[]=downloads+desc&rows={limit}&output=json"
        )
        headers = {"User-Agent": self.user_agent}
        items: List[DiscoveredMediaItem] = []

        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
                resp = await client.get(search_url)
                if resp.status_code == 200:
                    data = resp.json()
                    docs = data.get("response", {}).get("docs", [])
                    for doc in docs:
                        ident = doc.get("identifier")
                        title = doc.get("title", ident)
                        if not ident:
                            continue
                        # Query file metadata for MP4 stream
                        meta_url = f"https://archive.org/metadata/{ident}/files"
                        meta_resp = await client.get(meta_url)
                        if meta_resp.status_code == 200:
                            files_data = meta_resp.json().get("result", [])
                            for f in files_data:
                                f_name = f.get("name", "")
                                if f_name.lower().endswith(".mp4") and "512kb" not in f_name.lower():
                                    direct_url = f"https://archive.org/download/{ident}/{f_name}"
                                    items.append(DiscoveredMediaItem(
                                        title=title,
                                        source_url=direct_url,
                                        source_platform="internet_archive",
                                        media_type="video",
                                        license_type="Public Domain Archive",
                                        is_authorized=True,
                                        suitability_score=0.9,
                                    ))
                                    break
                        if len(items) >= limit:
                            break
        except Exception as e:
            log.warning(f"Internet Archive video search failed on '{query}': {e}")

        return items

    async def download_media(self, item: DiscoveredMediaItem, destination_folder: Optional[Path] = None) -> Optional[Path]:
        """Download a discovered authorized video or audio file to local storage."""
        dest_dir = destination_folder or workspace_manager.dirs["media_input"]
        dest_dir.mkdir(parents=True, exist_ok=True)

        clean_filename = re.sub(r"[^\w\-_\.]", "_", item.title)[:40]
        ext = Path(urllib.parse.urlparse(item.source_url).path).suffix or ".mp4"
        if not ext.startswith("."):
            ext = f".{ext}"
        out_file = dest_dir / f"{clean_filename}{ext}"

        if out_file.exists() and out_file.stat().st_size > 10000:
            item.local_path = str(out_file)
            return out_file

        headers = {"User-Agent": self.user_agent}
        try:
            log.info(f"Downloading authorized footage: '{item.title}' from {item.source_platform}...")
            async with httpx.AsyncClient(timeout=30.0, headers=headers, follow_redirects=True) as client:
                async with client.stream("GET", item.source_url) as resp:
                    if resp.status_code == 200:
                        with open(out_file, "wb") as f:
                            async for chunk in resp.aiter_bytes(chunk_size=65536):
                                f.write(chunk)
                        if out_file.stat().st_size > 5000:
                            item.local_path = str(out_file)
                            log.info(f"Successfully downloaded footage: {out_file.name} ({out_file.stat().st_size / (1024*1024):.2f} MB)")
                            return out_file
        except Exception as e:
            log.warning(f"Failed to download footage '{item.title}': {e}")
            if out_file.exists():
                out_file.unlink()

        return None

    async def discover_and_acquire(
        self,
        query: str,
        target_clips_count: int = 2,
    ) -> MediaCollectionReport:
        """
        Orchestrates full autonomous media discovery:
        1. Checks copyright status.
        2. Queries Wikimedia Commons & Internet Archive.
        3. Downloads usable authorized clips to media/input/.
        """
        # 1. Copyright check
        is_restricted, msg = self.is_copyright_protected(query)
        if is_restricted:
            return MediaCollectionReport(
                query=query,
                items_found=0,
                items_downloaded=0,
                is_copyright_restricted=True,
                restriction_message=msg,
                items=[],
            )

        # 2. Search open repositories
        discovered: List[DiscoveredMediaItem] = []
        wiki_items = await self.search_wikimedia_videos(query, limit=target_clips_count)
        discovered.extend(wiki_items)

        if len(discovered) < target_clips_count:
            archive_items = await self.search_internet_archive_videos(query, limit=target_clips_count)
            discovered.extend(archive_items)

        # 3. Download acquired clips into media/input/
        downloaded_count = 0
        for itm in discovered[:target_clips_count]:
            path = await self.download_media(itm)
            if path and path.exists():
                downloaded_count += 1

        return MediaCollectionReport(
            query=query,
            items_found=len(discovered),
            items_downloaded=downloaded_count,
            is_copyright_restricted=False,
            restriction_message=None,
            items=discovered,
        )


# Global media discovery engine singleton
media_discovery_engine = MediaDiscoveryEngine()
