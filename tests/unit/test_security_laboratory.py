"""
JARVIS AI — Phase 24 Ethical Cybersecurity Laboratory Test Suite
================================================================
Comprehensive unit tests covering:
1. AuthorizationGate (Affirmative consent, root drive rejection, scope verification)
2. SafeSystemAuditor (Non-destructive configuration and permission checks)
3. StaticFileAnalyzer (Shannon entropy, magic byte detection, masquerading flags, passive string triage)
4. VulnerablePracticeSandbox (MockSQLTarget, MockTemplateTarget, MockPathTarget defensive verification)
5. SecurityReportGenerator (Markdown report formatting and executive scorecards)
6. Agent #13 CybersecurityMonitoringAgent Laboratory Integration
"""

import os
import shutil
import tempfile
import secrets
import pytest

from services.security.laboratory import (
    AuthorizationGate,
    SafeSystemAuditor,
    StaticFileAnalyzer,
    MockSQLTarget,
    MockTemplateTarget,
    MockPathTarget,
    SecurityReportGenerator,
)
from services.security.laboratory.models import FindingSeverity
from core.orchestrator.agent_registry import agent_registry


@pytest.fixture
def temp_sandbox_dir():
    """Create a temporary sandbox directory for test assessments."""
    d = tempfile.mkdtemp(prefix="jarvis_lab_test_")
    yield os.path.realpath(d)
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)


# ── 1. Authorization Gate Tests ──────────────────────────────────────────────

def test_authorization_gate_workflow(temp_sandbox_dir):
    """Authorization grant, verification, and revocation."""
    gate = AuthorizationGate()

    # Unauthorized path should raise PermissionError
    with pytest.raises(PermissionError, match="Laboratory Access Denied"):
        gate.verify_target_authorized(temp_sandbox_dir)

    # Grant authorization
    scope = gate.request_authorization(temp_sandbox_dir, authorized_by="tester")
    assert scope.is_authorized is True
    assert gate.verify_target_authorized(temp_sandbox_dir) == temp_sandbox_dir

    # Revoke authorization
    assert gate.revoke_authorization(scope.scope_id) is True
    with pytest.raises(PermissionError):
        gate.verify_target_authorized(temp_sandbox_dir)


def test_authorization_gate_root_drive_protection():
    """Authorizing entire root drive must be blocked."""
    gate = AuthorizationGate()
    root_drive = "C:\\" if os.name == "nt" else "/"

    with pytest.raises(PermissionError, match="Cannot authorize entire root drive"):
        gate.request_authorization(root_drive)


# ── 2. Safe System Auditor Tests ─────────────────────────────────────────────

def test_safe_system_auditor_detects_secrets(temp_sandbox_dir):
    """Auditor detects unencrypted .env files and private keys in authorized paths."""
    gate = AuthorizationGate()
    gate.request_authorization(temp_sandbox_dir, authorized_by="tester")
    auditor = SafeSystemAuditor(gate=gate)

    # Place a mock exposed secret file
    secret_file = os.path.join(temp_sandbox_dir, ".env")
    with open(secret_file, "w") as f:
        f.write("DB_PASSWORD=super_secret_password\nAPI_KEY=xyz123\n")

    report = auditor.audit_target(temp_sandbox_dir, audit_network=False)
    assert report.total_findings >= 1
    assert any("Credential Management" in f.category for f in report.findings)
    assert any(f.severity == FindingSeverity.HIGH for f in report.findings)
    assert report.defensive_scorecard < 100.0


def test_safe_system_auditor_clean_directory(temp_sandbox_dir):
    """Clean directory with standard files receives a 100.0 defensive score."""
    gate = AuthorizationGate()
    gate.request_authorization(temp_sandbox_dir, authorized_by="tester")
    auditor = SafeSystemAuditor(gate=gate)

    # Clean file
    clean_file = os.path.join(temp_sandbox_dir, "notes.txt")
    with open(clean_file, "w") as f:
        f.write("Public project meeting notes.")

    report = auditor.audit_target(temp_sandbox_dir, audit_network=False)
    assert report.total_findings == 0
    assert report.defensive_scorecard == 100.0


# ── 3. Static File Quarantine Analyzer Tests ─────────────────────────────────

def test_static_file_analyzer_entropy(temp_sandbox_dir):
    """Calculates Shannon entropy: low for plaintext, high for random/packed data."""
    analyzer = StaticFileAnalyzer()

    # Low entropy file (repetitive characters)
    low_file = os.path.join(temp_sandbox_dir, "plain.txt")
    with open(low_file, "wb") as f:
        f.write(b"AAAAAAAAAA" * 100)

    low_rep = analyzer.analyze_file(low_file)
    assert low_rep.shannon_entropy < 3.0

    # High entropy file (cryptographically random bytes)
    high_file = os.path.join(temp_sandbox_dir, "random.bin")
    with open(high_file, "wb") as f:
        f.write(secrets.token_bytes(4096))

    high_rep = analyzer.analyze_file(high_file)
    assert high_rep.shannon_entropy > 7.2
    assert any("High Shannon Entropy" in ind for ind in high_rep.suspicious_indicators)


def test_static_file_analyzer_masquerading(temp_sandbox_dir):
    """Detects executable headers (MZ) masquerading with .pdf extension."""
    analyzer = StaticFileAnalyzer()

    fake_pdf = os.path.join(temp_sandbox_dir, "invoice.pdf")
    # Write Windows PE executable header (MZ) disguised as PDF
    with open(fake_pdf, "wb") as f:
        f.write(b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 100)

    rep = analyzer.analyze_file(fake_pdf)
    assert rep.is_masquerading is True
    assert rep.risk_assessment == "HIGH_RISK"
    assert rep.detected_mime_type == "application/x-dosexec"
    assert rep.declared_extension == ".pdf"


def test_static_file_analyzer_string_patterns(temp_sandbox_dir):
    """Detects suspicious PowerShell commands and private keys without executing."""
    analyzer = StaticFileAnalyzer()

    script_file = os.path.join(temp_sandbox_dir, "script.bat")
    with open(script_file, "w") as f:
        f.write("echo Starting...\npowershell -enc JAB4ACAAPQAgADEA\n")

    rep = analyzer.analyze_file(script_file)
    assert any("PowerShell invocation" in ind for ind in rep.suspicious_indicators)


# ── 4. In-Memory Vulnerable Practice Sandbox Tests ────────────────────────────

def test_mock_sql_target_defense():
    """MockSQLTarget verifies that parameterized queries defeat SQL injection."""
    target = MockSQLTarget()
    sqli_payload = "' OR '1'='1"

    # Vulnerable method leaks all users
    leaked = target.vulnerable_query(sqli_payload)
    assert len(leaked) == 3

    # Parameterized method blocks injection and returns 0 matches
    defended = target.secure_parameterized_query(sqli_payload)
    assert len(defended) == 0


def test_mock_template_target_defense():
    """MockTemplateTarget verifies HTML escaping defeats XSS scripts."""
    target = MockTemplateTarget()
    xss_payload = "<script>alert('xss')</script>"

    # Vulnerable method allows raw script tags
    vuln_html = target.vulnerable_render(xss_payload)
    assert "<script>" in vuln_html

    # Secure method entity-encodes script tags
    sec_html = target.secure_render(xss_payload)
    assert "<script>" not in sec_html
    assert "&lt;script&gt;" in sec_html


def test_mock_path_target_defense(temp_sandbox_dir):
    """MockPathTarget verifies that canonicalization defeats path traversal."""
    target = MockPathTarget(sandbox_root=temp_sandbox_dir)
    traversal_input = "../../../etc/passwd"

    # Vulnerable resolve escapes
    escaped = target.vulnerable_resolve(traversal_input)
    assert escaped != temp_sandbox_dir

    # Secure resolve raises PermissionError
    with pytest.raises(PermissionError, match="Directory traversal blocked"):
        target.secure_resolve(traversal_input)


# ── 5. Security Report Generator Tests ───────────────────────────────────────

def test_security_report_generator_formatting(temp_sandbox_dir):
    """Report generator formats complete markdown reports."""
    gen = SecurityReportGenerator()
    analyzer = StaticFileAnalyzer()

    dummy_file = os.path.join(temp_sandbox_dir, "test.txt")
    with open(dummy_file, "w") as f:
        f.write("Normal content")

    file_rep = analyzer.analyze_file(dummy_file)
    md = gen.format_file_analysis_report(file_rep)

    assert "Quarantine File Analysis Report" in md
    assert "SHA-256" in md
    assert "BENIGN" in md


# ── 6. Agent Integration Tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cybersecurity_agent_file_inspection(temp_sandbox_dir):
    """CybersecurityMonitoringAgent executes file inspection directives."""
    agent = agent_registry.get_agent("cybersecurity")
    assert agent is not None

    test_file = os.path.join(temp_sandbox_dir, "sample.txt")
    with open(test_file, "w") as f:
        f.write("Diagnostic text file for agent.")

    res = await agent.execute(f"inspect file {test_file}")
    assert res.success is True
    assert "Quarantine File Analysis Report" in res.summary
