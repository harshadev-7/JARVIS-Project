"""
JARVIS AI — Cybersecurity Intelligence Data Models
===================================================
Structured schema definitions for defensive security topics,
vulnerability taxonomies (CWE/CVE), CVSS scoring, and hardening checklists.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SecurityDomain(str, Enum):
    NETWORKING = "networking"
    PROTOCOLS = "protocols"
    SYSADMIN = "sysadmin"
    CRYPTOGRAPHY = "cryptography"
    WEB_SECURITY = "web_security"
    AUTHENTICATION = "authentication"
    VULNERABILITY = "vulnerability"
    MONITORING = "monitoring"
    DEFENSIVE_OPS = "defensive_ops"
    SECURE_DEV = "secure_dev"


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class VulnerabilityDef(BaseModel):
    """Common Weakness Enumeration (CWE) or vulnerability taxonomy entry."""
    cwe_id: str  # e.g., "CWE-89"
    name: str  # e.g., "SQL Injection"
    owasp_category: str  # e.g., "A03:2021-Injection"
    cvss_base_score: float  # e.g., 9.8
    severity: SeverityLevel
    description: str
    attack_vector: str
    impact: str
    mitigation_guidance: str
    secure_code_example: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class SecurityTopic(BaseModel):
    """Knowledge article covering defensive concepts, protocols, or architecture."""
    topic_id: str
    title: str
    domain: SecurityDomain
    summary: str
    technical_details: str
    threat_model: Optional[str] = None
    defensive_remediations: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    rfc_or_standards: List[str] = Field(default_factory=list)


class HardeningStep(BaseModel):
    step_id: str
    title: str
    category: str
    description: str
    verification_check: str
    recommended_config: str


class HardeningChecklist(BaseModel):
    target_system: str
    description: str
    steps: List[HardeningStep] = Field(default_factory=list)


class SecurityQueryResult(BaseModel):
    query: str
    matched_topics: List[SecurityTopic] = Field(default_factory=list)
    matched_vulnerabilities: List[VulnerabilityDef] = Field(default_factory=list)
    defensive_advisory: Optional[str] = None
    suggested_commands: List[str] = Field(default_factory=list)
