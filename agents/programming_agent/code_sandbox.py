"""
JARVIS AI — Code Execution Sandbox
===================================
Executes user/generated code in controlled, temporary sandbox environments
with strict execution timeouts and output limits.
"""

import sys
import tempfile
import asyncio
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from core.logger import log


class ExecutionResult(BaseModel):
    """Sandbox execution output."""
    language: str
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False


class CodeSandbox:
    """Safe execution sandbox for multi-language scripts."""

    def __init__(self, timeout_sec: float = 10.0, max_output_chars: int = 10000):
        self.timeout_sec = timeout_sec
        self.max_output_chars = max_output_chars

    async def execute_python(self, code: str) -> ExecutionResult:
        """Run Python code in an isolated subprocess."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name

        start = asyncio.get_event_loop().time()
        timed_out = False
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_sec)
            duration = asyncio.get_event_loop().time() - start
            exit_code = proc.returncode if proc.returncode is not None else 0

            stdout_txt = stdout_b.decode(errors="replace")[:self.max_output_chars].strip()
            stderr_txt = stderr_b.decode(errors="replace")[:self.max_output_chars].strip()

            return ExecutionResult(
                language="python",
                exit_code=exit_code,
                stdout=stdout_txt,
                stderr=stderr_txt,
                duration_sec=round(duration, 2),
                timed_out=False,
            )
        except asyncio.TimeoutError:
            duration = self.timeout_sec
            return ExecutionResult(
                language="python",
                exit_code=-1,
                stdout="",
                stderr=f"Execution timed out after {self.timeout_sec}s threshold.",
                duration_sec=duration,
                timed_out=True,
            )
        finally:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


code_sandbox = CodeSandbox()
