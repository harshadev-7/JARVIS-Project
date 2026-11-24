"""Unit tests for J.A.R.V.I.S. Phase 31 — Self-Code Analysis.
===========================================================
Tests Dimension 4 of the Self-Improvement System:
1. Static inspection of source code (AST-based, read-only)
2. Detecting bugs, bare excepts, and mutable default arguments
3. Detecting blocking synchronous calls inside async coroutines
4. Detecting security weaknesses (eval/exec, shell=True, hardcoded secrets)
5. Detecting duplicated code across files
6. Dependency analysis and circular import detection
7. Generating comprehensive code-health reports and grades
"""

import json
import pytest
from pathlib import Path

from core.self_evolution.self_code_analyzer import (
    SelfCodeAnalyzer,
    CodeIssue,
    CodeIssueType,
    IssueSeverity,
    DuplicateBlock,
    DependencyAnalysis,
    CodeQualityReport,
)


@pytest.fixture
def temp_analyzer(tmp_path):
    """Provides a SelfCodeAnalyzer rooted in a temporary workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evolution = tmp_path / "evolution"
    analyzer = SelfCodeAnalyzer(workspace_dir=workspace, evolution_dir=evolution)
    analyzer.initialize()
    return analyzer


def test_self_code_analyzer_initialization(temp_analyzer):
    """Verifies analyzer directories are created properly."""
    assert temp_analyzer.workspace_dir.exists()
    assert temp_analyzer.evolution_dir.exists()


def test_inspect_file_bug_detection(temp_analyzer):
    """Verifies detection of bare except handlers and mutable default arguments."""
    bad_code = (
        "def process_items(items=[]):\n"
        "    try:\n"
        "        val = items[0]\n"
        "    except:\n"
        "        val = None\n"
        "    return val\n"
    )
    test_file = temp_analyzer.workspace_dir / "bad_code.py"
    test_file.write_text(bad_code, encoding="utf-8")

    issues, metrics = temp_analyzer.inspect_file(test_file)

    assert metrics["loc"] >= 5
    assert metrics["functions"] == 1

    # Check mutable default issue
    mut_issues = [i for i in issues if "mutable" in i.title.lower()]
    assert len(mut_issues) == 1
    assert mut_issues[0].severity == IssueSeverity.HIGH.value
    assert mut_issues[0].line_number == 1

    # Check bare except issue
    except_issues = [i for i in issues if "bare exception" in i.title.lower()]
    assert len(except_issues) == 1
    assert except_issues[0].severity == IssueSeverity.MEDIUM.value
    assert except_issues[0].line_number == 4


def test_inspect_file_security_detection(temp_analyzer):
    """Verifies static detection of security risks (eval, shell=True, md5)."""
    sec_code = (
        "import subprocess\n"
        "import hashlib\n\n"
        "def run_command(cmd):\n"
        "    res = eval(cmd)\n"
        "    p = subprocess.Popen(cmd, shell=True)\n"
        "    h = hashlib.md5(b'test').hexdigest()\n"
        "    return res\n"
    )
    test_file = temp_analyzer.workspace_dir / "sec_code.py"
    test_file.write_text(sec_code, encoding="utf-8")

    issues, _ = temp_analyzer.inspect_file(test_file)
    sec_issues = [i for i in issues if i.issue_type == CodeIssueType.SECURITY.value]

    assert len(sec_issues) >= 2
    # Verify eval() flagged as CRITICAL
    eval_issues = [i for i in sec_issues if "eval" in i.description.lower()]
    assert len(eval_issues) == 1
    assert eval_issues[0].severity == IssueSeverity.CRITICAL.value

    # Verify shell=True flagged as HIGH
    shell_issues = [i for i in sec_issues if "shell=true" in i.description.lower()]
    assert len(shell_issues) == 1
    assert shell_issues[0].severity == IssueSeverity.HIGH.value


def test_inspect_file_async_blocking_detection(temp_analyzer):
    """Verifies detection of synchronous time.sleep() inside async coroutines."""
    async_code = (
        "import asyncio\n"
        "import time\n\n"
        "async def fetch_data():\n"
        "    time.sleep(2)  # Blocking call in async loop\n"
        "    return {'status': 'ok'}\n"
    )
    test_file = temp_analyzer.workspace_dir / "async_blocking.py"
    test_file.write_text(async_code, encoding="utf-8")

    issues, _ = temp_analyzer.inspect_file(test_file)
    blocking_issues = [i for i in issues if "blocking" in i.title.lower()]

    assert len(blocking_issues) == 1
    assert blocking_issues[0].severity == IssueSeverity.HIGH.value
    assert "time.sleep" in blocking_issues[0].description
    assert blocking_issues[0].line_number == 5


def test_detect_duplicated_code(temp_analyzer):
    """Verifies detection of identical code blocks across separate files."""
    shared_block = (
        "    # Data normalization helper\n"
        "    cleaned = text.strip().lower()\n"
        "    tokens = cleaned.split()\n"
        "    filtered = [t for t in tokens if len(t) > 2]\n"
        "    result = ' '.join(filtered)\n"
        "    return result\n"
    )

    file_a = temp_analyzer.workspace_dir / "module_a.py"
    file_a.write_text(f"def normalize_v1(text):\n{shared_block}", encoding="utf-8")

    file_b = temp_analyzer.workspace_dir / "module_b.py"
    file_b.write_text(f"def clean_input(text):\n{shared_block}", encoding="utf-8")

    duplicates = temp_analyzer.detect_duplicated_code([file_a, file_b], min_lines=5)

    assert len(duplicates) >= 1
    dup = duplicates[0]
    assert isinstance(dup, DuplicateBlock)
    assert "module_a.py" in [dup.file_a, dup.file_b]
    assert "module_b.py" in [dup.file_a, dup.file_b]
    assert dup.line_count >= 5


def test_analyze_dependencies_and_circular_imports(temp_analyzer):
    """Verifies dependency extraction and detection of circular imports."""
    file_1 = temp_analyzer.workspace_dir / "service_alpha.py"
    file_1.write_text("import service_beta\n\ndef run_alpha(): return 1\n", encoding="utf-8")

    file_2 = temp_analyzer.workspace_dir / "service_beta.py"
    file_2.write_text("import service_alpha\n\ndef run_beta(): return 2\n", encoding="utf-8")

    dep_analysis = temp_analyzer.analyze_dependencies([file_1, file_2])

    assert isinstance(dep_analysis, DependencyAnalysis)
    assert dep_analysis.total_imports >= 2
    # Circular loop between service_alpha and service_beta should be detected
    assert len(dep_analysis.circular_dependencies) >= 1
    loop = dep_analysis.circular_dependencies[0]
    assert any("service_alpha" in m for m in loop)
    assert any("service_beta" in m for m in loop)


def test_run_full_codebase_analysis(temp_analyzer):
    """Verifies full static analysis report generation and health score computation."""
    # Write clean code file
    clean_code = (
        "class SafeCalculator:\n"
        "    def add(self, a: int, b: int) -> int:\n"
        "        return a + b\n\n"
        "    def multiply(self, a: int, b: int) -> int:\n"
        "        return a * b\n"
    )
    file_clean = temp_analyzer.workspace_dir / "safe_calc.py"
    file_clean.write_text(clean_code, encoding="utf-8")

    report = temp_analyzer.run_full_codebase_analysis(target_dirs=["."])

    assert isinstance(report, CodeQualityReport)
    assert report.total_files_scanned >= 1
    assert report.total_lines_of_code >= 5
    assert report.total_classes >= 1
    assert report.total_functions >= 2
    assert report.code_health_score >= 90.0
    assert report.grade in ["A+", "A"]
    assert "JARVIS Code Health" in report.summary

    # Verify report persistence
    assert temp_analyzer.reports_file.exists()
    with open(temp_analyzer.reports_file, "r", encoding="utf-8") as f:
        history = json.load(f)
    assert len(history) >= 1


def test_generate_report_markdown(temp_analyzer):
    """Verifies markdown report generation for CLI and HUD display."""
    clean_code = "def greet(name: str) -> str: return f'Hello, {name}'\n"
    f = temp_analyzer.workspace_dir / "greet.py"
    f.write_text(clean_code, encoding="utf-8")

    md = temp_analyzer.generate_report_markdown()

    assert "J.A.R.V.I.S. Self-Code Analysis Report (Phase 31)" in md
    assert "Code Health Score" in md
    assert "Files Scanned" in md
    assert "Key Recommendations" in md
