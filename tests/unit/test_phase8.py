"""
JARVIS AI — Phase 8 Test Suite
================================
Tests for Programming and Software Development Intelligence:
- Code Static Analysis, Function & Class Extraction
- Security Smell Detection (Hardcoded Secrets & Dangerous eval)
- Sandboxed Subprocess Execution with Timeout Protection
- Automated Traceback Diagnosis and Debugging
- FastAPI Programming Endpoints
"""

import pytest
from fastapi.testclient import TestClient

from agents.programming_agent.code_analyzer import code_analyzer
from agents.programming_agent.code_sandbox import code_sandbox
from agents.programming_agent.code_debugger import code_debugger
from api.app import app


# ── 1. Code Analyzer Tests ──────────────────────────────────────────────────

def test_code_analyzer_structure():
    """Analyzer should extract functions and classes from valid Python code."""
    sample_code = """
class DataPipeline:
    def __init__(self):
        pass

    def process_records(self, items):
        return [i * 2 for i in items]
    """
    report = code_analyzer.analyze(sample_code, language="python")
    assert report.is_valid_syntax is True
    assert "DataPipeline" in report.classes_found
    assert "process_records" in report.functions_found
    assert report.security_score == 1.0


def test_code_analyzer_security_flags():
    """Analyzer should detect hardcoded credentials and dangerous eval."""
    insecure_code = """
api_key = 'super_secret_token_12345678'
eval('malicious_input()')
    """
    report = code_analyzer.analyze(insecure_code, language="python")
    assert len(report.issues) >= 2
    assert report.security_score < 1.0
    rules = [i.rule_id for i in report.issues]
    assert "SEC-001" in rules
    assert "SEC-002" in rules


def test_code_analyzer_syntax_error():
    """Syntax error should be captured with line number."""
    bad_code = "def broken_func(\n    return 42"
    report = code_analyzer.analyze(bad_code, language="python")
    assert report.is_valid_syntax is False
    assert any(i.severity == "error" for i in report.issues)


# ── 2. Code Sandbox Tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_code_sandbox_execution():
    """Sandbox should execute script cleanly and capture stdout."""
    code = "print('COMPUTATION_COMPLETE: ' + str(21 * 2))"
    res = await code_sandbox.execute_python(code)
    assert res.exit_code == 0
    assert "COMPUTATION_COMPLETE: 42" in res.stdout
    assert res.timed_out is False


@pytest.mark.asyncio
async def test_code_sandbox_timeout():
    """Sandbox should terminate hanging/infinite loops safely."""
    infinite_code = "import time\nwhile True:\n    time.sleep(0.1)"
    # Set brief 1-second timeout for testing
    code_sandbox.timeout_sec = 1.0
    res = await code_sandbox.execute_python(infinite_code)
    assert res.timed_out is True
    assert "timed out" in res.stderr
    code_sandbox.timeout_sec = 10.0  # Reset default


# ── 3. Code Debugger Tests ──────────────────────────────────────────────────

def test_code_debugger_traceback():
    """Debugger should diagnose ZeroDivisionError and recommend guard."""
    tb = """
Traceback (most recent call last):
  File "calculator.py", line 14, in compute_ratio
    result = total / count
ZeroDivisionError: division by zero
    """
    report = code_debugger.diagnose_traceback(tb)
    assert report.error_type == "ZeroDivisionError"
    assert report.failing_line == 14
    assert "denominator" in report.root_cause_explanation.lower() or "zero" in report.root_cause_explanation.lower()
    assert "guard" in report.suggested_fix.lower() or "!=" in report.suggested_fix


# ── 4. FastAPI Programming Endpoints Tests ──────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


def test_api_programming_analyze(client):
    """POST /programming/analyze should return analysis report."""
    res = client.post(
        "/programming/analyze",
        json={"code": "def hello():\n    return 'world'", "language": "python"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid_syntax"] is True
    assert "hello" in data["functions_found"]


def test_api_programming_execute(client):
    """POST /programming/execute should run in sandbox."""
    res = client.post(
        "/programming/execute",
        json={"code": "print('JARVIS_CORE_OK')"},
    )
    assert res.status_code == 200
    assert "JARVIS_CORE_OK" in res.json()["stdout"]


def test_api_programming_debug(client):
    """POST /programming/debug should analyze error."""
    res = client.post(
        "/programming/debug",
        json={"traceback": "IndexError: list index out of range"},
    )
    assert res.status_code == 200
    assert res.json()["error_type"] == "IndexError"
