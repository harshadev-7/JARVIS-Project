"""
JARVIS AI — Programming & Software Intelligence API Routes
===========================================================
Endpoints for static analysis, security linting, sandboxed execution,
and automated error debugging.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agents.programming_agent.code_analyzer import code_analyzer, CodeAnalysisReport
from agents.programming_agent.code_sandbox import code_sandbox, ExecutionResult
from agents.programming_agent.code_debugger import code_debugger, DebugReport

router = APIRouter(prefix="/programming", tags=["Programming Intelligence"])


class AnalyzeCodeRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Source code text")
    language: str = Field("python", description="Programming language (e.g. python, javascript, sql)")


class ExecuteCodeRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Python script to run in sandbox")


class DebugCodeRequest(BaseModel):
    traceback: str = Field(..., min_length=5, description="Error traceback or compiler output")
    source_code: Optional[str] = Field(None, description="Optional failing source code")


@router.post("/analyze", response_model=CodeAnalysisReport, summary="Static syntax, complexity & security analysis")
async def analyze_code(request: AnalyzeCodeRequest):
    return code_analyzer.analyze(request.code, language=request.language)


@router.post("/execute", response_model=ExecutionResult, summary="Execute code in safe sandboxed environment")
async def execute_code(request: ExecuteCodeRequest):
    return await code_sandbox.execute_python(request.code)


@router.post("/debug", response_model=DebugReport, summary="Diagnose stack trace and formulate fix")
async def debug_code(request: DebugCodeRequest):
    return code_debugger.diagnose_traceback(request.traceback, source_code=request.source_code)
