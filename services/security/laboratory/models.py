"""
JARVIS AI — Ethical Cybersecurity Laboratory Data Models
=========================================================
Structured schemas for authorization scopes, vulnerability findings,
quarantine file analysis reports, and defensive audit summaries.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class FindingSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AssessmentScope(BaseModel):
    """Defines explicit boundaries and authorization for a security evaluation."""
    scope_id: str
    target_path: str
    authorized_by: str
    authorized_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_authorized: bool = False
    allow_network_audit: bool = True
    allow_permission_audit: bool = True
    allow_file_inspection: bool = True


class SecurityVulnerabilityFinding(BaseModel):
    """An individual security weakness identified during read-only evaluation."""
    finding_id: str
    title: str
    category: str
    severity: FindingSeverity
    cvss_score: float
    description: str
    affected_target: str
    evidence: str
    remediation_guidance: str
    cwe_id: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class FileAnalysisReport(BaseModel):
    """Results of safe, passive static file quarantine inspection."""
    file_path: str
    file_size_bytes: int
    sha256_hash: str
    sha1_hash: str
    md5_hash: str
    shannon_entropy: float
    detected_mime_type: str
    declared_extension: str
    is_masquerading: bool = False
    suspicious_indicators: List[str] = Field(default_factory=list)
    risk_assessment: str = "BENIGN"  # "BENIGN", "SUSPICIOUS", "ANOMALOUS", "HIGH_RISK"
    defensive_recommendations: List[str] = Field(default_factory=list)


class LaboratoryAuditReport(BaseModel):
    """Consolidated defensive report for an authorized system assessment."""
    assessment_id: str
    target_system: str
    conducted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_sec: float = 0.0
    total_findings: int = 0
    severity_counts: Dict[str, int] = Field(default_factory=dict)
    findings: List[SecurityVulnerabilityFinding] = Field(default_factory=list)
    executive_summary: str = ""
    defensive_scorecard: float = 100.0  # 0 to 100 defense score
