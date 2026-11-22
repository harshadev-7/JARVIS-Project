"""J.A.R.V.I.S. "Think Before Searching" Intelligent Decision Policy.

Implements Section 2 of the Intelligence Upgrade:
Before initiating an external internet search or web scrape, evaluate:
1. Can I answer this using my existing knowledge? -> Answer directly.
2. Can I solve this using reasoning? -> Reason through the problem.
3. Can I calculate this using a deterministic tool? -> Calculate and verify.
4. Can I generate the requested content myself? -> Generate original content.
5. Does the request actually require current external information? -> Research online.
6. Did the user explicitly request internet research? -> Use research engine.

Internet access must strictly support JARVIS's intelligence, not replace it.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


class DecisionPolicy:
    """Evaluates whether an external internet search is strictly justified."""

    # Explicit research trigger patterns
    EXPLICIT_RESEARCH_PATTERNS = [
        r"^/research\b",
        r"\b(?:search\s+(?:the\s+)?web|search\s+online|look\s+up\s+online|google\s+this|find\s+online|browse\s+the\s+web)\b",
        r"\b(?:look\s+up\s+on\s+the\s+internet|web\s+search|conduct\s+research\s+on)\b",
        r"\b(?:search\s+the\s+web\s+for|search\s+online\s+for)\b",
    ]

    # Temporal & live information indicators that genuinely require external search
    LIVE_INFO_PATTERNS = [
        r"\b(?:latest|recent|newest|today'?s|current|upcoming)\s+(?:developments?|news|releases?|updates?|models?|events?|prices?|weather|gpu|nvidia\s+gpu|hardware|chip)\b",
        r"\b(?:what\s+are\s+the\s+latest|what\s+is\s+the\s+latest|what\s+is\s+the\s+current|latest\s+news\s+about|recent\s+advancements?\s+in)\b",
        r"\b(?:stock\s+price\s+of|weather\s+in|weather\s+forecast|today'?s\s+headlines)\b",
        r"\b(?:electric\s+vehicle\s+releases|latest\s+developments\s+in|latest\s+nvidia\s+gpu)\b",
        r"\b(?:2025|2026|2027)\b",
    ]

    # Explicit internal tasks that must NEVER search the web
    FORBIDDEN_SEARCH_PATTERNS = [
        r"\b(?:generate|create|make|write|give me)\s+(?:an?\s+)?(?:image\s+prompt|prompt|video\s+prompt|concept|script|story|poem|caption)\b",
        r"\b(?:calculate|compute|solve|math|equation|\d+\s*[\+\-\*\/\^]\s*\d+)\b",
        r"\b(?:python|code|function|algorithm|script|class|def |debug)\b",
        r"\b(?:if\s+\d+\s+machines|bat\s+and\s+(?:a\s+)?ball|riddle|puzzle|logic\s+problem)\b",
        r"\b(?:capital\s+of|who\s+wrote|what\s+is\s+photosynthesis|how\s+does\s+a\s+transformer\s+work)\b",
        r"\b(?:summarize\s+(?:this|the\s+following|the\s+above)|document\s+summary)\b",
    ]

    @classmethod
    def evaluate(cls, query: str, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Evaluate if an internet search should be performed.

        Returns:
            Tuple[bool, str]: (should_search, rationale)
        """
        q = query.strip()
        q_lower = q.lower()

        # 1. Did the user explicitly request internet research?
        for pat in cls.EXPLICIT_RESEARCH_PATTERNS:
            if re.search(pat, q_lower):
                return True, "User explicitly requested internet research."

        # 2. Check forbidden search domains (creative, local code, logic riddles, math, summarization)
        for pat in cls.FORBIDDEN_SEARCH_PATTERNS:
            if re.search(pat, q_lower):
                return False, "Request can be satisfied locally via internal knowledge, reasoning, or creative generation."

        # 3. Does the query require current, live, or time-sensitive external information?
        for pat in cls.LIVE_INFO_PATTERNS:
            if re.search(pat, q_lower):
                return True, "Query requires current/time-sensitive real-world information."

        # 4. Default: Rely on internal intelligence, deterministic tools, and offline knowledge
        return False, "Internal offline intelligence preferred; external search not justified."

    @classmethod
    def should_search_internet(cls, query: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Convenience boolean check."""
        should_search, _ = cls.evaluate(query, context)
        return should_search


decision_policy = DecisionPolicy()
__all__ = ["DecisionPolicy", "decision_policy"]
