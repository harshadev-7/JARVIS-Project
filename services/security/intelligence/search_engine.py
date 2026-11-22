"""
JARVIS AI — Cybersecurity Inverted Search & Advisory Engine
===========================================================
High-speed local search index for defensive cybersecurity intelligence.
Provides:
- Sub-5ms token-weighted inverted index retrieval.
- CWE vulnerability lookup and CVSS scoring explanation.
- Defensive remediation and system hardening checklist generation.
- Ethical Guardrail: Deflects weaponized exploit requests into defensive mitigations.
"""

import re
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

from services.security.intelligence.models import (
    SecurityTopic,
    SecurityDomain,
    VulnerabilityDef,
    HardeningChecklist,
    SecurityQueryResult,
    SeverityLevel,
)
from services.security.intelligence.knowledge_base import (
    KNOWLEDGE_TOPICS,
    VULNERABILITY_CATALOG,
    LINUX_HARDENING_CHECKLIST,
)
from core.logger import log


# Offensive request triggers to deflect defensively
OFFENSIVE_KEYWORDS = [
    r"\b(write|create|give\s+me|generate)\s+(?:an?\s+)?(?:exploit|payload|shellcode|malware|keylogger|backdoor)\b",
    r"\b(how\s+to\s+hack|steal\s+passwords|bypass\s+authentication|crack\s+password|ddos\s+attack)\b",
    r"\b(sql\s+injection\s+payload|xss\s+payload\s+to\s+steal)\b",
]


STOP_WORDS = {"how", "to", "the", "is", "an", "a", "and", "or", "of", "in", "for", "on", "with", "give", "me", "from", "by", "what"}


class CybersecurityIntelligenceEngine:
    """Indexed local query engine for defensive cybersecurity knowledge."""

    def __init__(self):
        self.topics = KNOWLEDGE_TOPICS
        self.vulnerabilities = VULNERABILITY_CATALOG
        self._inverted_index: Dict[str, Set[str]] = defaultdict(set)  # token -> set of topic_ids / cwe_ids
        self._topics_by_id: Dict[str, SecurityTopic] = {}
        self._vulns_by_cwe: Dict[str, VulnerabilityDef] = {}
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Convert string to normalized search tokens."""
        clean = re.sub(r"[^\w\s-]", " ", text.lower())
        tokens = [t.strip() for t in clean.split() if len(t.strip()) >= 2 and t.strip() not in STOP_WORDS]
        return tokens

    def _build_index(self):
        """Index all topics and CWE vulnerabilities into in-memory inverted index."""
        # Index topics
        for topic in self.topics:
            self._topics_by_id[topic.topic_id] = topic
            content = f"{topic.title} {topic.domain.value} {topic.summary} {topic.technical_details} {' '.join(topic.tags)}"
            for token in self._tokenize(content):
                self._inverted_index[token].add(f"topic:{topic.topic_id}")

        # Index vulnerabilities
        for vuln in self.vulnerabilities:
            clean_cwe = vuln.cwe_id.lower()
            self._vulns_by_cwe[clean_cwe] = vuln
            self._vulns_by_cwe[clean_cwe.replace("-", "")] = vuln
            content = f"{vuln.cwe_id} {vuln.name} {vuln.owasp_category} {vuln.description} {vuln.attack_vector} {vuln.mitigation_guidance}"
            for token in self._tokenize(content):
                self._inverted_index[token].add(f"vuln:{clean_cwe}")

        log.info(f"CybersecurityIntelligenceEngine indexed {len(self.topics)} topics and {len(self.vulnerabilities)} CWEs.")

    def is_offensive_query(self, query: str) -> bool:
        """Detect if user query is attempting to solicit weaponized exploits or attacks."""
        lower = query.lower()
        for pat in OFFENSIVE_KEYWORDS:
            if re.search(pat, lower):
                return True
        return False

    def query(
        self,
        search_query: str,
        domain_filter: Optional[SecurityDomain] = None,
        top_k: int = 3,
    ) -> SecurityQueryResult:
        """
        Execute high-speed ranked search across defensive topics and CWE definitions.
        """
        is_offensive = self.is_offensive_query(search_query)
        tokens = self._tokenize(search_query)

        topic_scores: Dict[str, int] = defaultdict(int)
        vuln_scores: Dict[str, int] = defaultdict(int)

        # Direct CWE match (e.g. "CWE-89" or "89")
        cwe_match = re.search(r"\b(?:cwe[-_]?)?(\d+)\b", search_query.lower())
        if cwe_match:
            candidate = f"cwe-{cwe_match.group(1)}"
            if candidate in self._vulns_by_cwe:
                vuln_scores[candidate] += 50

        # Inverted index scoring
        for token in tokens:
            matches = self._inverted_index.get(token, set())
            for entry in matches:
                if entry.startswith("topic:"):
                    tid = entry.split(":", 1)[1]
                    topic = self._topics_by_id.get(tid)
                    if topic:
                        if domain_filter and topic.domain != domain_filter:
                            continue
                        # Weight title & tag matches higher
                        if token in [t.lower() for t in topic.tags] or token in topic.title.lower():
                            topic_scores[tid] += 5
                        else:
                            topic_scores[tid] += 1
                elif entry.startswith("vuln:"):
                    cwe = entry.split(":", 1)[1]
                    vuln = self._vulns_by_cwe.get(cwe)
                    if vuln and (token in vuln.name.lower() or token in vuln.cwe_id.lower()):
                        vuln_scores[cwe] += 15
                    else:
                        vuln_scores[cwe] += 3

        # Rank and collect top topics
        ranked_tids = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        matched_topics = [self._topics_by_id[tid] for tid, _ in ranked_tids]

        # Rank and collect top vulnerabilities
        ranked_cwes = sorted(vuln_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        matched_vulns = [self._vulns_by_cwe[cwe] for cwe, _ in ranked_cwes]

        # Synthesize advisory
        top_topic_score = ranked_tids[0][1] if ranked_tids else 0
        top_vuln_score = ranked_cwes[0][1] if ranked_cwes else 0
        is_vuln_query = any(w in search_query.lower() for w in [
            "injection", "xss", "csrf", "ssrf", "traversal", "deserialization",
            "cwe", "cve", "vulnerability", "exploit", "sqli", "weakness"
        ])
        advisory = self._synthesize_advisory(
            search_query,
            matched_topics,
            matched_vulns,
            is_offensive,
            prefer_vuln=bool(is_vuln_query or (top_vuln_score > top_topic_score)),
        )

        return SecurityQueryResult(
            query=search_query,
            matched_topics=matched_topics,
            matched_vulnerabilities=matched_vulns,
            defensive_advisory=advisory,
            suggested_commands=[
                f"/cwe {v.cwe_id}" for v in matched_vulns
            ] + [f"/secinfo {t.domain.value}" for t in matched_topics[:2]],
        )

    def get_vulnerability(self, cwe_id_or_num: str) -> Optional[VulnerabilityDef]:
        """Lookup specific CWE vulnerability by ID or number."""
        clean = cwe_id_or_num.lower().strip()
        if not clean.startswith("cwe-"):
            clean = f"cwe-{clean}"
        return self._vulns_by_cwe.get(clean)

    def get_hardening_checklist(self, target: str = "linux") -> HardeningChecklist:
        """Retrieve pre-built operating system hardening checklists."""
        return LINUX_HARDENING_CHECKLIST

    def _synthesize_advisory(
        self,
        query: str,
        topics: List[SecurityTopic],
        vulns: List[VulnerabilityDef],
        is_offensive: bool,
        prefer_vuln: bool = True,
    ) -> str:
        """Formulate a comprehensive defensive advisory text."""
        sections = []

        if is_offensive:
            sections.append(
                "🛡️ **Defensive Policy Notice:** In adherence to ethical security standards, "
                "JARVIS does not generate attack payloads or offensive exploitation scripts. "
                "Below is the comprehensive technical analysis, threat model, and defensive remediation guidance."
            )

        if prefer_vuln and vulns:
            v = vulns[0]
            sections.append(
                f"### 🛑 Vulnerability Analysis: {v.cwe_id} — {v.name}\n"
                f"- **OWASP Category:** `{v.owasp_category}`\n"
                f"- **CVSS Base Score:** **{v.cvss_base_score}** ({v.severity.value})\n"
                f"- **Attack Vector:** {v.attack_vector}\n"
                f"- **Impact:** {v.impact}\n\n"
                f"**Defensive Remediation:**\n{v.mitigation_guidance}\n"
            )
            if v.secure_code_example:
                sections.append(f"**Secure Code Pattern:**\n```python\n{v.secure_code_example}\n```")

        elif topics:
            t = topics[0]
            sections.append(
                f"### 🛡️ Defensive Architecture: {t.title} ({t.domain.value.upper()})\n"
                f"{t.summary}\n\n"
                f"**Technical Overview:**\n{t.technical_details}\n\n"
                f"**Defensive Countermeasures:**\n" +
                "\n".join([f"• {rem}" for rem in t.defensive_remediations])
            )

        elif vulns:
            v = vulns[0]
            sections.append(
                f"### 🛑 Vulnerability Analysis: {v.cwe_id} — {v.name}\n"
                f"- **OWASP Category:** `{v.owasp_category}`\n"
                f"- **CVSS Base Score:** **{v.cvss_base_score}** ({v.severity.value})\n"
                f"- **Attack Vector:** {v.attack_vector}\n"
                f"- **Impact:** {v.impact}\n\n"
                f"**Defensive Remediation:**\n{v.mitigation_guidance}\n"
            )
            if v.secure_code_example:
                sections.append(f"**Secure Code Pattern:**\n```python\n{v.secure_code_example}\n```")

        else:
            sections.append(
                f"No direct defensive catalog matches found for '{query}'. "
                f"Try searching for specific concepts (e.g. 'DNSSEC', 'TLS 1.3', 'SQLi', 'CWE-89', 'OAuth2')."
            )

        return "\n\n".join(sections)


cybersecurity_intelligence_engine = CybersecurityIntelligenceEngine()
