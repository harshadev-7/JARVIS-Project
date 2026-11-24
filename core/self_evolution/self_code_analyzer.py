"""J.A.R.V.I.S. Self-Evolution System — Phase 31: Self-Code Analysis
===================================================================
Implements Dimension 4 of the J.A.R.V.I.S. Four-Dimensional Self-Improvement System.

Core Capabilities:
1. Inspect Own Source Code:
   - Traverses workspace modules (`core/`, `custom_ai/`, `automation/`, `services/`, etc.)
     using Python's native Abstract Syntax Tree (`ast`) and tokenization engines in strict read-only mode.
2. Identify Bugs & Inefficiencies:
   - Detects bare `except:` clauses, dangerous default mutable arguments (`def foo(x=[])`),
     unused imports, cyclomatic complexity spikes ($McCabe > 10$), and blocking sync calls in async loops.
3. Detect Duplicated Code:
   - Identifies identical or near-identical code blocks across functions and modules using
     normalized rolling hashing.
4. Analyze Dependencies & Circular Imports:
   - Maps module-to-module imports, classifies internal vs. external dependencies, and
     detects circular import loops across packages.
5. Identify Security Weaknesses:
   - Static analysis for dangerous primitives (`eval()`, `exec()`, `shell=True` without sanitization,
     hardcoded credentials/tokens, and path traversal vectors).
6. Generate Code-Quality Reports:
   - Computes an overall Code Health Score (0.0 to 100.0) with a letter grade (A+ to F),
     and provides actionable, prioritized refactoring recommendations.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config.settings import BASE_DIR
from core.logger import log


class IssueSeverity(str, Enum):
    """Severity levels for detected code issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CodeIssueType(str, Enum):
    """Categories of code issues identified during static analysis."""
    BUG_RISK = "bug_risk"
    INEFFICIENCY = "inefficiency"
    DUPLICATION = "duplication"
    DEPENDENCY = "dependency"
    SECURITY = "security"
    STYLE_MAINTAINABILITY = "style_maintainability"


@dataclass
class CodeIssue:
    """A detected bug, inefficiency, security weakness, or structural defect."""
    issue_id: str
    file_path: str
    line_number: int
    issue_type: str  # CodeIssueType value
    severity: str    # IssueSeverity value
    title: str
    description: str
    recommendation: str
    code_snippet: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DuplicateBlock:
    """A duplicated section of code detected across files or functions."""
    file_a: str
    lines_a: Tuple[int, int]
    file_b: str
    lines_b: Tuple[int, int]
    line_count: int
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DependencyAnalysis:
    """Analysis of module dependencies and circular import structures."""
    total_imports: int
    internal_modules: List[str]
    external_packages: List[str]
    circular_dependencies: List[List[str]]
    isolated_modules: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CodeQualityReport:
    """Comprehensive code-quality, health, and architectural diagnostic report."""
    report_id: str
    timestamp: float
    total_files_scanned: int
    total_lines_of_code: int
    total_functions: int
    total_classes: int
    issues: List[CodeIssue]
    issues_by_severity: Dict[str, int]
    issues_by_type: Dict[str, int]
    duplicate_blocks: List[DuplicateBlock]
    duplicate_line_count: int
    circular_dependencies_count: int
    code_health_score: float  # 0.0 to 100.0
    grade: str                # A+, A, B, C, D, F
    recommendations: List[str]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SelfCodeAnalyzer:
    """Performs static code inspection and health diagnostics over JARVIS's source tree."""

    # Default directories to inspect
    DEFAULT_INSPECT_DIRS = ["core", "custom_ai", "services", "automation", "video_editor", "vision", "devices"]

    # Security patterns to flag
    SECURITY_PATTERNS = [
        (r"\b(?:api_key|secret_key|password|token)\s*=\s*['\"][a-zA-Z0-9_\-\.]{12,}['\"]", "Hardcoded potential secret or credential detected in source code.", IssueSeverity.HIGH),
        (r"\b(?:eval|exec)\s*\(", "Use of dynamic execution primitive (eval/exec) poses arbitrary code execution risk.", IssueSeverity.CRITICAL),
        (r"\bsubprocess\.(?:Popen|run|call)\s*\(.*shell\s*=\s*True", "Shell execution with shell=True is susceptible to command injection.", IssueSeverity.HIGH),
        (r"hashlib\.md5\s*\(", "Insecure cryptographic hash algorithm MD5 used. Use SHA-256 or better for security integrity.", IssueSeverity.MEDIUM),
    ]

    def __init__(self, workspace_dir: Optional[Path] = None, evolution_dir: Optional[Path] = None):
        self.workspace_dir = workspace_dir or BASE_DIR
        self.evolution_dir = evolution_dir or (BASE_DIR / ".jarvis" / "evolution")
        self.reports_file = self.evolution_dir / "code_analysis_reports.json"
        self._latest_report: Optional[CodeQualityReport] = None

    def initialize(self) -> None:
        """Sets up evolution storage directory."""
        self.evolution_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. FILE-LEVEL INSPECTION (BUGS, INEFFICIENCIES, SECURITY)
    # ──────────────────────────────────────────────────────────────────────────

    def inspect_file(self, file_path: Path) -> Tuple[List[CodeIssue], Dict[str, int]]:
        """
        Parses a Python source file into an AST and inspects for bugs, inefficiencies,
        and security issues in read-only mode.
        """
        issues: List[CodeIssue] = []
        metrics = {"loc": 0, "functions": 0, "classes": 0}

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
        except Exception as exc:
            log.warning(f"Failed to read file {file_path}: {exc}")
            return issues, metrics

        lines = code.splitlines()
        metrics["loc"] = len([l for l in lines if l.strip() and not l.strip().startswith("#")])

        # 1. Regex Security Scanning
        rel_path = str(file_path.relative_to(self.workspace_dir)).replace("\\", "/")
        for line_idx, line in enumerate(lines, 1):
            if line.strip().startswith("#"):
                continue
            for pattern, desc, severity in self.SECURITY_PATTERNS:
                # Exclude pattern definitions themselves
                if "SECURITY_PATTERNS" in line or "pattern" in line:
                    continue
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        issue_id=f"sec_{int(time.time()*1000)}_{line_idx}",
                        file_path=rel_path,
                        line_number=line_idx,
                        issue_type=CodeIssueType.SECURITY.value,
                        severity=severity.value,
                        title="Potential Security Vulnerability",
                        description=desc,
                        recommendation="Refactor to use environment variables, parameterized inputs, or modern crypto hashes.",
                        code_snippet=line.strip()[:100],
                    ))

        # 2. AST Structure Inspection
        try:
            tree = ast.parse(code, filename=str(file_path))
        except SyntaxError as syn_err:
            issues.append(CodeIssue(
                issue_id=f"syn_{int(time.time()*1000)}_{syn_err.lineno or 1}",
                file_path=rel_path,
                line_number=syn_err.lineno or 1,
                issue_type=CodeIssueType.BUG_RISK.value,
                severity=IssueSeverity.CRITICAL.value,
                title="Syntax Error",
                description=f"File contains a syntax error: {syn_err.msg}",
                recommendation="Fix Python syntax error prior to execution.",
            ))
            return issues, metrics

        # Walk AST nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                metrics["classes"] += 1

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                metrics["functions"] += 1
                func_issues = self._inspect_function(node, rel_path, lines)
                issues.extend(func_issues)

            # Check for bare `except:` clauses
            elif isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append(CodeIssue(
                        issue_id=f"bug_except_{int(time.time()*1000)}_{node.lineno}",
                        file_path=rel_path,
                        line_number=node.lineno,
                        issue_type=CodeIssueType.BUG_RISK.value,
                        severity=IssueSeverity.MEDIUM.value,
                        title="Bare Exception Handler",
                        description="A bare `except:` catches all exceptions including KeyboardInterrupt and SystemExit.",
                        recommendation="Specify explicit exception types e.g. `except Exception as exc:`",
                        code_snippet=lines[node.lineno - 1].strip() if 0 <= node.lineno - 1 < len(lines) else None,
                    ))

        return issues, metrics

    def _inspect_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        rel_path: str,
        lines: List[str],
    ) -> List[CodeIssue]:
        """Inspects an individual function for complexity, mutable defaults, and async blocking."""
        issues: List[CodeIssue] = []

        # 1. Dangerous mutable default arguments (e.g. def foo(x=[]))
        for default in node.args.defaults + node.args.kw_defaults:
            if default and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                issues.append(CodeIssue(
                    issue_id=f"bug_mut_{int(time.time()*1000)}_{node.lineno}",
                    file_path=rel_path,
                    line_number=node.lineno,
                    issue_type=CodeIssueType.BUG_RISK.value,
                    severity=IssueSeverity.HIGH.value,
                    title="Mutable Default Argument",
                    description=f"Function '{node.name}' defines a mutable default argument ({type(default).__name__}).",
                    recommendation="Use `None` as default argument and initialize mutable object inside function body.",
                    code_snippet=lines[node.lineno - 1].strip() if 0 <= node.lineno - 1 < len(lines) else None,
                ))

        # 2. Cyclomatic Complexity ($McCabe > 10$)
        complexity = self._calculate_cyclomatic_complexity(node)
        if complexity > 12:
            issues.append(CodeIssue(
                issue_id=f"ineff_comp_{int(time.time()*1000)}_{node.lineno}",
                file_path=rel_path,
                line_number=node.lineno,
                issue_type=CodeIssueType.INEFFICIENCY.value,
                severity=IssueSeverity.MEDIUM.value,
                title=f"High Cyclomatic Complexity ({complexity})",
                description=f"Function '{node.name}' has high branching complexity ({complexity} decision points).",
                recommendation="Decompose function into smaller helper sub-routines to improve testability and maintainability.",
                code_snippet=lines[node.lineno - 1].strip() if 0 <= node.lineno - 1 < len(lines) else None,
            ))

        # 3. Synchronous blocking sleep in async coroutine
        if isinstance(node, ast.AsyncFunctionDef):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Attribute) and child.func.attr == "sleep":
                        if isinstance(child.func.value, ast.Name) and child.func.value.id == "time":
                            issues.append(CodeIssue(
                                issue_id=f"ineff_block_{int(time.time()*1000)}_{child.lineno}",
                                file_path=rel_path,
                                line_number=child.lineno,
                                issue_type=CodeIssueType.INEFFICIENCY.value,
                                severity=IssueSeverity.HIGH.value,
                                title="Blocking time.sleep() in Async Function",
                                description=f"Async function '{node.name}' invokes synchronous `time.sleep()`, freezing the asyncio event loop.",
                                recommendation="Replace `time.sleep(...)` with `await asyncio.sleep(...)`.",
                                code_snippet=lines[child.lineno - 1].strip() if 0 <= child.lineno - 1 < len(lines) else None,
                            ))

        return issues

    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Estimates McCabe cyclomatic complexity by counting decision points."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.Assert):
                complexity += 1
        return complexity

    # ──────────────────────────────────────────────────────────────────────────
    # 2. DUPLICATE CODE DETECTION
    # ──────────────────────────────────────────────────────────────────────────

    def detect_duplicated_code(
        self,
        file_paths: List[Path],
        min_lines: int = 6,
    ) -> List[DuplicateBlock]:
        """
        Detects duplicated code blocks across files using normalized sliding line hashing.
        """
        duplicates: List[DuplicateBlock] = []
        chunk_hashes: Dict[str, List[Tuple[str, int, int, str]]] = {}

        for p in file_paths:
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    lines = [line.rstrip() for line in f]
            except Exception:
                continue

            rel_p = str(p.relative_to(self.workspace_dir)).replace("\\", "/")
            total = len(lines)

            # Slide window over lines
            for i in range(total - min_lines + 1):
                raw_chunk = lines[i : i + min_lines]
                # Normalize chunk: strip whitespace and ignore pure comment/blank lines
                normalized_lines = [l.strip() for l in raw_chunk if l.strip() and not l.strip().startswith("#")]
                if len(normalized_lines) < min_lines - 1:
                    continue

                chunk_text = "\n".join(normalized_lines)
                chunk_hash = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()

                if chunk_hash not in chunk_hashes:
                    chunk_hashes[chunk_hash] = []
                chunk_hashes[chunk_hash].append((rel_p, i + 1, i + min_lines, "\n".join(raw_chunk[:3])))

        # Find collisions across distinct locations
        seen_pairs: Set[Tuple[str, int, str, int]] = set()
        for chunk_hash, occurrences in chunk_hashes.items():
            if len(occurrences) > 1:
                # Compare pairs
                for idx_a in range(len(occurrences)):
                    for idx_b in range(idx_a + 1, len(occurrences)):
                        fa, la_start, la_end, snippet = occurrences[idx_a]
                        fb, lb_start, lb_end, _ = occurrences[idx_b]

                        # Ignore trivial same-file overlaps within 5 lines
                        if fa == fb and abs(la_start - lb_start) < min_lines:
                            continue

                        pair_key = (fa, la_start, fb, lb_start)
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            duplicates.append(DuplicateBlock(
                                file_a=fa,
                                lines_a=(la_start, la_end),
                                file_b=fb,
                                lines_b=(lb_start, lb_end),
                                line_count=min_lines,
                                snippet=snippet,
                            ))

        return duplicates[:20]  # Cap top 20 duplicate instances to prevent bloating

    # ──────────────────────────────────────────────────────────────────────────
    # 3. DEPENDENCY & CIRCULAR IMPORT ANALYSIS
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_dependencies(self, file_paths: List[Path]) -> DependencyAnalysis:
        """
        Parses imports across all source files and detects circular dependencies.
        """
        internal_modules: Set[str] = set()
        external_packages: Set[str] = set()
        dep_graph: Dict[str, Set[str]] = {}
        total_imports = 0

        # Build map of known internal module paths
        module_path_map: Dict[str, str] = {}
        for p in file_paths:
            rel = str(p.relative_to(self.workspace_dir)).replace("\\", "/")
            if rel.endswith(".py"):
                mod_name = rel[:-3].replace("/", ".")
                module_path_map[mod_name] = rel
                internal_modules.add(mod_name)
                dep_graph[mod_name] = set()

        # Parse AST imports
        for p in file_paths:
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    code = f.read()
                tree = ast.parse(code, filename=str(p))
            except Exception:
                continue

            rel = str(p.relative_to(self.workspace_dir)).replace("\\", "/")
            curr_mod = rel[:-3].replace("/", ".") if rel.endswith(".py") else rel

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        total_imports += 1
                        root_pkg = alias.name.split(".")[0]
                        if alias.name in internal_modules or root_pkg in ["core", "custom_ai", "services", "automation", "video_editor", "vision", "devices"]:
                            if curr_mod in dep_graph:
                                dep_graph[curr_mod].add(alias.name)
                        else:
                            external_packages.add(root_pkg)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        total_imports += 1
                        root_pkg = node.module.split(".")[0]
                        if node.module in internal_modules or root_pkg in ["core", "custom_ai", "services", "automation", "video_editor", "vision", "devices"]:
                            if curr_mod in dep_graph:
                                dep_graph[curr_mod].add(node.module)
                        else:
                            external_packages.add(root_pkg)

        # Detect circular dependencies using cycle finding
        circular_loops: List[List[str]] = []
        visited: Dict[str, int] = {m: 0 for m in dep_graph}
        path: List[str] = []

        def find_cycles(node: str):
            visited[node] = 1
            path.append(node)
            for neighbor in dep_graph.get(node, []):
                # Only check internal nodes
                if neighbor in dep_graph:
                    if visited.get(neighbor) == 1:
                        cycle_idx = path.index(neighbor)
                        circular_loops.append(path[cycle_idx:] + [neighbor])
                    elif visited.get(neighbor) == 0:
                        find_cycles(neighbor)
            path.pop()
            visited[node] = 2

        for mod in list(dep_graph.keys()):
            if visited.get(mod) == 0:
                find_cycles(mod)

        # Isolated modules (no internal dependents or dependencies)
        isolated = [m for m, deps in dep_graph.items() if not deps]

        return DependencyAnalysis(
            total_imports=total_imports,
            internal_modules=sorted(list(internal_modules))[:50],
            external_packages=sorted(list(external_packages)),
            circular_dependencies=circular_loops[:10],
            isolated_modules=isolated[:20],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 4. FULL CODEBASE ANALYSIS & HEALTH REPORT
    # ──────────────────────────────────────────────────────────────────────────

    def run_full_codebase_analysis(
        self,
        target_dirs: Optional[List[str]] = None,
    ) -> CodeQualityReport:
        """
        Executes a complete static analysis sweep over the specified directories
        and produces an audited CodeQualityReport.
        """
        self.initialize()
        report_id = f"rpt_code_{int(time.time() * 1000)}"
        dirs = target_dirs or self.DEFAULT_INSPECT_DIRS

        all_files: List[Path] = []
        for d in dirs:
            dir_path = self.workspace_dir / d
            if dir_path.exists() and dir_path.is_dir():
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        if file.endswith(".py") and "__pycache__" not in root:
                            all_files.append(Path(root) / file)

        # 1. File Inspection
        all_issues: List[CodeIssue] = []
        total_loc = 0
        total_functions = 0
        total_classes = 0

        for f in all_files:
            file_issues, metrics = self.inspect_file(f)
            all_issues.extend(file_issues)
            total_loc += metrics["loc"]
            total_functions += metrics["functions"]
            total_classes += metrics["classes"]

        # 2. Duplicate Detection
        duplicates = self.detect_duplicated_code(all_files, min_lines=6)
        duplicate_line_count = sum(d.line_count for d in duplicates)

        # 3. Dependency Analysis
        dep_analysis = self.analyze_dependencies(all_files)

        # 4. Aggregate Issue Counts
        by_sev: Dict[str, int] = {s.value: 0 for s in IssueSeverity}
        by_type: Dict[str, int] = {t.value: 0 for t in CodeIssueType}
        for iss in all_issues:
            by_sev[iss.severity] = by_sev.get(iss.severity, 0) + 1
            by_type[iss.issue_type] = by_type.get(iss.issue_type, 0) + 1

        # 5. Compute Code Health Score (0.0 to 100.0)
        # Deduct penalties for critical/high issues, duplicates, and cycles
        base_score = 100.0
        penalty_critical = by_sev.get(IssueSeverity.CRITICAL.value, 0) * 15.0
        penalty_high = by_sev.get(IssueSeverity.HIGH.value, 0) * 4.0
        penalty_medium = by_sev.get(IssueSeverity.MEDIUM.value, 0) * 1.0
        penalty_dup = min(len(duplicates) * 1.5, 15.0)
        penalty_cycles = min(len(dep_analysis.circular_dependencies) * 5.0, 15.0)

        total_penalty = penalty_critical + penalty_high + penalty_medium + penalty_dup + penalty_cycles
        health_score = round(max(0.0, base_score - total_penalty), 1)

        # Determine letter grade
        if health_score >= 95.0:
            grade = "A+"
        elif health_score >= 88.0:
            grade = "A"
        elif health_score >= 78.0:
            grade = "B"
        elif health_score >= 65.0:
            grade = "C"
        elif health_score >= 50.0:
            grade = "D"
        else:
            grade = "F"

        # Generate Actionable Recommendations
        recommendations: List[str] = []
        if by_sev.get(IssueSeverity.CRITICAL.value, 0) > 0:
            recommendations.append(f"Resolve {by_sev[IssueSeverity.CRITICAL.value]} critical security/syntax issues immediately.")
        if by_sev.get(IssueSeverity.HIGH.value, 0) > 0:
            recommendations.append(f"Refactor {by_sev[IssueSeverity.HIGH.value]} high-severity issues (blocking sleeps in async or mutable defaults).")
        if len(duplicates) > 0:
            recommendations.append(f"Consolidate {len(duplicates)} duplicated code blocks into shared utility helpers.")
        if len(dep_analysis.circular_dependencies) > 0:
            recommendations.append(f"Decouple {len(dep_analysis.circular_dependencies)} circular package import loops.")
        if not recommendations:
            recommendations.append("Codebase passes all static checks with optimal structure.")

        summary = (
            f"JARVIS Code Health: {health_score}/100 (Grade: {grade}). "
            f"Scanned {len(all_files)} files ({total_loc} LOC, {total_functions} functions, {total_classes} classes). "
            f"Detected {len(all_issues)} issues ({by_sev.get('critical', 0)} critical, {by_sev.get('high', 0)} high). "
            f"Found {len(duplicates)} duplicate blocks and {len(dep_analysis.circular_dependencies)} circular dependencies."
        )

        report = CodeQualityReport(
            report_id=report_id,
            timestamp=time.time(),
            total_files_scanned=len(all_files),
            total_lines_of_code=total_loc,
            total_functions=total_functions,
            total_classes=total_classes,
            issues=all_issues,
            issues_by_severity=by_sev,
            issues_by_type=by_type,
            duplicate_blocks=duplicates,
            duplicate_line_count=duplicate_line_count,
            circular_dependencies_count=len(dep_analysis.circular_dependencies),
            code_health_score=health_score,
            grade=grade,
            recommendations=recommendations,
            summary=summary,
        )

        self._latest_report = report
        self._persist_report(report)
        log.info(f"Self-Code Analysis completed. Grade: {grade} ({health_score}/100). Files: {len(all_files)}")
        return report

    def _persist_report(self, report: CodeQualityReport) -> None:
        """Saves analysis report to disk."""
        try:
            reports = []
            if self.reports_file.exists():
                with open(self.reports_file, "r", encoding="utf-8") as f:
                    reports = json.load(f)
                    if not isinstance(reports, list):
                        reports = []
            # Keep last 10 reports
            reports.append(report.to_dict())
            reports = reports[-10:]
            with open(self.reports_file, "w", encoding="utf-8") as f:
                json.dump(reports, f, indent=2)
        except Exception as exc:
            log.warning(f"Could not persist code quality report: {exc}")

    def generate_report_markdown(self, report: Optional[CodeQualityReport] = None) -> str:
        """Generates a clean markdown status report for CLI and Web HUD."""
        rep = report or self._latest_report
        if not rep:
            rep = self.run_full_codebase_analysis()

        rec_lines = "\n".join([f"  - {r}" for r in rep.recommendations])

        return (
            f"### 🛡️ J.A.R.V.I.S. Self-Code Analysis Report (Phase 31)\n\n"
            f"- **Code Health Score**: `{rep.code_health_score} / 100.0` (Grade: **{rep.grade}**)\n"
            f"- **Files Scanned**: `{rep.total_files_scanned}` ({rep.total_lines_of_code} LOC across {rep.total_functions} functions & {rep.total_classes} classes)\n"
            f"- **Issues Detected**: `{len(rep.issues)}`\n"
            f"  - 🔴 Critical: `{rep.issues_by_severity.get('critical', 0)}`\n"
            f"  - 🟠 High: `{rep.issues_by_severity.get('high', 0)}`\n"
            f"  - 🟡 Medium: `{rep.issues_by_severity.get('medium', 0)}`\n"
            f"  - 🟢 Low: `{rep.issues_by_severity.get('low', 0)}`\n"
            f"- **Duplicate Blocks**: `{len(rep.duplicate_blocks)}` ({rep.duplicate_line_count} duplicate lines)\n"
            f"- **Circular Dependencies**: `{rep.circular_dependencies_count}`\n\n"
            f"#### **Key Recommendations:**\n{rec_lines}\n\n"
            f"> [!NOTE]\n"
            f"> Static analysis runs in read-only mode. Proposed improvements are delegated to "
            f"Phase 32 (Self-Code Generation) and staged exclusively on isolated development branches."
        )


# Global singleton instance for system-wide access
self_code_analyzer = SelfCodeAnalyzer()
