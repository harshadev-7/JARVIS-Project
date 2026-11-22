"""
JARVIS AI — Phase 23 Cybersecurity Intelligence Test Suite
==========================================================
Tests covering:
1. Curated Knowledge Catalog (all 10 domains & CWE definitions)
2. Cybersecurity Inverted Search & Indexing Engine
3. CWE Vulnerability Lookup & CVSS Scoring Analysis
4. Defensive Ethical Guardrails (Exploit solicitation deflection)
5. Operating System Hardening Checklist Generation
6. CybersecurityMonitoringAgent (Agent #13) Integration
7. Intent Router Classification & Brain Integration
"""

import pytest
from services.security.intelligence import (
    cybersecurity_intelligence_engine,
    CybersecurityIntelligenceEngine,
    SecurityDomain,
    SeverityLevel,
    KNOWLEDGE_TOPICS,
    VULNERABILITY_CATALOG,
    LINUX_HARDENING_CHECKLIST,
)
from core.orchestrator.agent_registry import agent_registry
from core.brain.router import IntentRouter, IntentType


# ── 1. Knowledge Catalog Tests ───────────────────────────────────────────────

def test_knowledge_topics_comprehensive_coverage():
    """Knowledge catalog must contain structured topics across core security domains."""
    assert len(KNOWLEDGE_TOPICS) >= 10
    domains = {t.domain for t in KNOWLEDGE_TOPICS}

    # Verify coverage across key domains
    assert SecurityDomain.NETWORKING in domains
    assert SecurityDomain.PROTOCOLS in domains
    assert SecurityDomain.SYSADMIN in domains
    assert SecurityDomain.CRYPTOGRAPHY in domains
    assert SecurityDomain.WEB_SECURITY in domains
    assert SecurityDomain.AUTHENTICATION in domains
    assert SecurityDomain.VULNERABILITY in domains
    assert SecurityDomain.MONITORING in domains
    assert SecurityDomain.DEFENSIVE_OPS in domains
    assert SecurityDomain.SECURE_DEV in domains

    for topic in KNOWLEDGE_TOPICS:
        assert len(topic.title) > 0
        assert len(topic.technical_details) > 20
        assert len(topic.defensive_remediations) > 0


def test_vulnerability_catalog_definitions():
    """CWE definitions must have valid CVSS scores, descriptions, and remediations."""
    assert len(VULNERABILITY_CATALOG) >= 8

    cwe_ids = [v.cwe_id for v in VULNERABILITY_CATALOG]
    assert "CWE-89" in cwe_ids  # SQLi
    assert "CWE-79" in cwe_ids  # XSS
    assert "CWE-22" in cwe_ids  # Path Traversal
    assert "CWE-78" in cwe_ids  # Command Injection
    assert "CWE-918" in cwe_ids # SSRF

    for vuln in VULNERABILITY_CATALOG:
        assert 0.0 <= vuln.cvss_base_score <= 10.0
        assert vuln.severity in (SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL)
        assert len(vuln.mitigation_guidance) > 10


# ── 2. Inverted Search Engine Tests ──────────────────────────────────────────

def test_search_engine_cwe_lookup():
    """Searching for 'CWE-89' or 'sql injection' must match CWE-89 definition."""
    engine = CybersecurityIntelligenceEngine()

    # Direct CWE lookup
    vuln = engine.get_vulnerability("CWE-89")
    assert vuln is not None
    assert "SQL" in vuln.name
    assert vuln.cvss_base_score >= 9.0

    # Query search
    res = engine.query("how to prevent sql injection")
    assert len(res.matched_vulnerabilities) > 0
    assert res.matched_vulnerabilities[0].cwe_id == "CWE-89"
    assert "Parameterized" in res.defensive_advisory or "parameterized" in res.defensive_advisory


def test_search_engine_protocol_topics():
    """Searching for TLS 1.3, DNSSEC, or SYN flood returns appropriate topics."""
    engine = CybersecurityIntelligenceEngine()

    res_tls = engine.query("TLS 1.3 handshake and cipher suites")
    assert any("TLS 1.3" in t.title for t in res_tls.matched_topics)

    res_dns = engine.query("DNS cache poisoning defense")
    assert any("DNS" in t.title for t in res_dns.matched_topics)

    res_syn = engine.query("SYN flood attack mitigation")
    assert any("SYN" in t.title for t in res_syn.matched_topics)


def test_search_engine_domain_filtering():
    """Domain filter must restrict matched topics strictly to the requested domain."""
    engine = CybersecurityIntelligenceEngine()
    res = engine.query("architecture", domain_filter=SecurityDomain.SYSADMIN)

    for topic in res.matched_topics:
        assert topic.domain == SecurityDomain.SYSADMIN


# ── 3. Defensive Ethical Guardrail Tests ─────────────────────────────────────

def test_offensive_query_defensive_deflection():
    """Attacking prompts must trigger the defensive ethical notice and mitigation guidance."""
    engine = CybersecurityIntelligenceEngine()

    offensive_prompt = "generate an exploit payload to bypass authentication with sql injection"
    assert engine.is_offensive_query(offensive_prompt) is True

    res = engine.query(offensive_prompt)
    assert "Defensive Policy Notice" in res.defensive_advisory
    assert "Defensive Remediation" in res.defensive_advisory


# ── 4. System Hardening Checklist Tests ──────────────────────────────────────

def test_linux_hardening_checklist():
    """Hardening checklist should provide concrete steps with verification commands."""
    engine = CybersecurityIntelligenceEngine()
    chk = engine.get_hardening_checklist("linux")

    assert chk.target_system.startswith("Linux")
    assert len(chk.steps) >= 4

    categories = [s.category for s in chk.steps]
    assert "SSH Hardening" in categories
    assert "Network Defense" in categories


# ── 5. Agent & Router Integration Tests ──────────────────────────────────────

@pytest.mark.asyncio
async def test_cybersecurity_agent_execution():
    """Agent #13 CybersecurityMonitoringAgent executes security queries cleanly."""
    agent = agent_registry.get_agent("cybersecurity")
    assert agent is not None

    # Query 1: Topic query
    res = await agent.execute("explain zero trust architecture and webauthn")
    assert res.success is True
    assert "Zero Trust" in res.summary

    # Query 2: Hardening checklist
    chk_res = await agent.execute("give me a checklist for linux server hardening")
    assert chk_res.success is True
    assert "Hardening Checklist" in chk_res.summary


def test_intent_router_cybersecurity_classification():
    """Router must classify security questions under IntentType.CYBERSECURITY."""
    router = IntentRouter()

    d1 = router.route("/secinfo tls 1.3")
    assert d1.intent == IntentType.CYBERSECURITY
    assert d1.agent_name == "CybersecurityMonitoringAgent"

    d2 = router.route("/cwe 89")
    assert d2.intent == IntentType.CYBERSECURITY
    assert d2.agent_name == "CybersecurityMonitoringAgent"

    d3 = router.route("how do we prevent cross-site scripting and xss in web applications?")
    assert d3.intent == IntentType.CYBERSECURITY
