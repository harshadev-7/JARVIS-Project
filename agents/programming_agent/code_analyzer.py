"""
JARVIS AI — Code Analyzer & Linter
===================================
Inspects code for syntax validity, security vulnerabilities (hardcoded secrets,
SQL injections, dangerous eval), and structure metrics.
"""

import ast
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class CodeIssue(BaseModel):
    """An issue identified in code."""
    line: int
    severity: str  # "error", "warning", "security"
    message: str
    rule_id: str


class CodeAnalysisReport(BaseModel):
    """Static analysis results."""
    language: str
    is_valid_syntax: bool
    total_lines: int
    functions_found: List[str] = Field(default_factory=list)
    classes_found: List[str] = Field(default_factory=list)
    issues: List[CodeIssue] = Field(default_factory=list)
    security_score: float = Field(default=1.0, ge=0.0, le=1.0)


class CodeAnalyzer:
    """Static analyzer for multiple programming languages."""

    # Security smell regexes
    SECURITY_PATTERNS = [
        (r"(?i)(api_key|secret|password|token)\s*=\s*['\"][a-zA-Z0-9_\-]{8,}['\"]", "SEC-001", "Possible hardcoded secret or credential"),
        (r"\beval\(", "SEC-002", "Insecure use of eval() detected"),
        (r"\bexec\(", "SEC-003", "Dangerous use of exec() detected"),
        (r"(?i)select\s+.*\s+from\s+.*%\s*\(", "SEC-004", "Possible unparameterized SQL formatting"),
    ]

    def analyze(self, code: str, language: str = "python") -> CodeAnalysisReport:
        """Analyze code for syntax, structure, and security smells."""
        lang = language.lower().strip()
        lines = code.splitlines()
        issues: List[CodeIssue] = []

        # 1. Regex Security Scan
        for idx, line in enumerate(lines, 1):
            for pattern, rule_id, msg in self.SECURITY_PATTERNS:
                if re.search(pattern, line):
                    issues.append(CodeIssue(line=idx, severity="security", message=msg, rule_id=rule_id))

        # 2. Python AST Parsing
        functions: List[str] = []
        classes: List[str] = []
        is_valid = True

        if lang == "python":
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        functions.append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        classes.append(node.name)
            except SyntaxError as e:
                is_valid = False
                issues.append(
                    CodeIssue(
                        line=e.lineno or 1,
                        severity="error",
                        message=f"Syntax Error: {e.msg}",
                        rule_id="SYN-001",
                    )
                )

        sec_issues = [i for i in issues if i.severity == "security"]
        sec_score = max(0.0, 1.0 - (len(sec_issues) * 0.25))

        return CodeAnalysisReport(
            language=lang,
            is_valid_syntax=is_valid,
            total_lines=len(lines),
            functions_found=functions,
            classes_found=classes,
            issues=issues,
            security_score=round(sec_score, 2),
        )


code_analyzer = CodeAnalyzer()
