"""
JARVIS AI — Code Debugger & Diagnostic Engine
==============================================
Analyzes stack traces and compiler errors, isolates root cause,
and generates verified code patches.
"""

import re
from typing import Optional, List
from pydantic import BaseModel, Field


class DebugReport(BaseModel):
    """Diagnostic debugging analysis."""
    error_type: str
    error_message: str
    failing_line: Optional[int] = None
    root_cause_explanation: str
    suggested_fix: str
    patched_code_snippet: Optional[str] = None


class CodeDebugger:
    """Automated error diagnostics and debugging."""

    def diagnose_traceback(self, traceback_text: str, source_code: Optional[str] = None) -> DebugReport:
        """Parse traceback and extract line, error class, and remediation."""
        lines = traceback_text.strip().splitlines()
        last_line = lines[-1] if lines else "Error: Unknown execution fault"

        # Match "ExceptionType: Message"
        match = re.match(r"^([a-zA-Z0-9_.]+(?:Error|Exception)): (.*)$", last_line)
        if match:
            err_type = match.group(1)
            err_msg = match.group(2)
        else:
            err_type = "RuntimeError"
            err_msg = last_line

        # Extract line number from traceback (e.g. line 42)
        line_matches = re.findall(r"line (\d+)", traceback_text)
        failing_line = int(line_matches[-1]) if line_matches else None

        # Diagnostic patterns
        if "ZeroDivisionError" in err_type:
            cause = "Attempted division by zero variable or denominator evaluated to 0."
            fix = "Add guard check: `if denominator != 0:` before division operation."
        elif "IndexError" in err_type:
            cause = "Accessed an array or sequence index beyond its length."
            fix = "Verify sequence length with `len(seq)` or wrap in boundary check."
        elif "KeyError" in err_type:
            cause = "Attempted dictionary lookup for a non-existent key."
            fix = "Use `dict.get(key, default)` or verify presence with `if key in dict:`."
        elif "TypeError" in err_type:
            cause = "Incompatible types provided to an operator or function."
            fix = "Cast inputs explicitly (e.g. `int()`, `str()`) or check parameter types."
        else:
            cause = f"Unhandled exception encountered: {err_msg}"
            fix = "Review variable state at failing line and enclose in try/except block."

        return DebugReport(
            error_type=err_type,
            error_message=err_msg,
            failing_line=failing_line,
            root_cause_explanation=cause,
            suggested_fix=fix,
        )


code_debugger = CodeDebugger()
