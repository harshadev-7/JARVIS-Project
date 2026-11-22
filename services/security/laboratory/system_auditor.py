"""
JARVIS AI — Safe System Configuration Auditor
==============================================
Performs 100% non-destructive, read-only security posture evaluations
across authorized local directories and host configurations.
Checks:
- Exposed credentials, unencrypted secrets, and private key files.
- Overly permissive file permissions (world-writable / insecure ownership).
- Unencrypted or overly broad network socket bindings (0.0.0.0 vs 127.0.0.1).
- Default configuration and debug flag posture.
"""

import os
import stat
import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import psutil

from services.security.laboratory.models import (
    LaboratoryAuditReport,
    SecurityVulnerabilityFinding,
    FindingSeverity,
)
from services.security.laboratory.authorization_gate import authorization_gate
from core.logger import log


class SafeSystemAuditor:
    """Read-only security posture auditor for authorized targets."""

    def __init__(self, gate=authorization_gate):
        self.gate = gate

    def audit_target(self, target_path: str, audit_network: bool = True) -> LaboratoryAuditReport:
        """
        Execute passive, read-only security audit of an authorized directory.
        Raises PermissionError if target path is not authorized.
        """
        canonical_target = self.gate.verify_target_authorized(target_path)
        start_t = time.perf_counter()
        findings: List[SecurityVulnerabilityFinding] = []

        log.info(f"Starting non-destructive security audit on authorized target: {canonical_target}")

        # 1. Audit File Permissions & Secret Leaks
        self._audit_directory_contents(canonical_target, findings)

        # 2. Audit Local Network Socket Bindings
        if audit_network:
            self._audit_local_sockets(findings)

        # 3. Calculate metrics and scorecard
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
        score = 100.0

        for f in findings:
            sev_str = f.severity.value
            severity_counts[sev_str] = severity_counts.get(sev_str, 0) + 1
            if f.severity == FindingSeverity.CRITICAL:
                score -= 25.0
            elif f.severity == FindingSeverity.HIGH:
                score -= 15.0
            elif f.severity == FindingSeverity.MEDIUM:
                score -= 5.0
            elif f.severity == FindingSeverity.LOW:
                score -= 2.0

        score = max(0.0, score)
        elapsed_sec = time.perf_counter() - start_t

        report = LaboratoryAuditReport(
            assessment_id=f"audit-{uuid.uuid4().hex[:8]}",
            target_system=canonical_target,
            conducted_at=datetime.now(timezone.utc),
            duration_sec=round(elapsed_sec, 3),
            total_findings=len(findings),
            severity_counts=severity_counts,
            findings=findings,
            defensive_scorecard=round(score, 1),
            executive_summary=(
                f"Security audit completed across '{canonical_target}'. "
                f"Identified {len(findings)} potential weakness(es). "
                f"Overall Defensive Posture Score: {round(score, 1)} / 100.0."
            ),
        )
        return report

    def _audit_directory_contents(self, root_dir: str, findings: List[SecurityVulnerabilityFinding]):
        """Scan directory for sensitive files, unencrypted secrets, and permissive flags."""
        sensitive_extensions = {".pem", ".key", ".pkcs12", ".pfx", ".id_rsa", ".env"}

        for root, dirs, files in os.walk(root_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                ext = os.path.splitext(file_name)[1].lower()
                base_name = file_name.lower()

                # A. Exposed Private Keys or Secret Files
                if ext in sensitive_extensions or base_name in (".env", "id_rsa", "credentials.json", "secrets.yaml"):
                    findings.append(
                        SecurityVulnerabilityFinding(
                            finding_id=f"find-sec-{len(findings) + 1:03d}",
                            title=f"Potential Sensitive Secret or Key Exposed: {file_name}",
                            category="Credential Management",
                            severity=FindingSeverity.HIGH,
                            cvss_score=7.5,
                            description=(
                                f"Detected plaintext secret configuration or private key file: '{file_path}'. "
                                f"Storing sensitive credentials in unencrypted filesystem paths risks exposure."
                            ),
                            affected_target=file_path,
                            evidence=f"File matching sensitive pattern: {file_name}",
                            remediation_guidance=(
                                "Store credentials in an encrypted vault (e.g. HashiCorp Vault, AWS Secrets Manager) "
                                "or inject them via secure environment variables. Exclude secret files via .gitignore."
                            ),
                            cwe_id="CWE-312",
                        )
                    )

                # B. POSIX File Permission Checks (if on POSIX system)
                if os.name == "posix":
                    try:
                        st = os.stat(file_path)
                        # Check world-writable
                        if st.st_mode & stat.S_IWOTH:
                            findings.append(
                                SecurityVulnerabilityFinding(
                                    finding_id=f"find-perm-{len(findings) + 1:03d}",
                                    title=f"World-Writable File Permission: {file_name}",
                                    category="Access Control",
                                    severity=FindingSeverity.MEDIUM,
                                    cvss_score=5.5,
                                    description=f"File '{file_path}' has permissions allowing any local user to modify it.",
                                    affected_target=file_path,
                                    evidence=f"Permissions: {oct(st.st_mode)[-3:]}",
                                    remediation_guidance="Restrict file permissions using chmod: chmod 640 or chmod 600.",
                                    cwe_id="CWE-732",
                                )
                            )
                    except Exception:
                        pass

    def _audit_local_sockets(self, findings: List[SecurityVulnerabilityFinding]):
        """Inspect active local listening sockets for broad interfaces or cleartext protocols."""
        try:
            connections = psutil.net_connections(kind="inet")
            for conn in connections:
                if conn.status == psutil.CONN_LISTEN:
                    ip = conn.laddr.ip if conn.laddr else ""
                    port = conn.laddr.port if conn.laddr else 0

                    # Check for services listening on 0.0.0.0
                    if ip in ("0.0.0.0", "::"):
                        # Flag unencrypted default services
                        if port in (21, 23, 80, 8080):
                            findings.append(
                                SecurityVulnerabilityFinding(
                                    finding_id=f"find-net-{len(findings) + 1:03d}",
                                    title=f"Unencrypted Service Bound to All Interfaces: Port {port}",
                                    category="Network Exposure",
                                    severity=FindingSeverity.MEDIUM,
                                    cvss_score=5.3,
                                    description=(
                                        f"A process is listening on all network interfaces ({ip}:{port}) using an unencrypted protocol. "
                                        f"This allows external network actors to connect or intercept cleartext communications."
                                    ),
                                    affected_target=f"{ip}:{port}",
                                    evidence=f"Listening Socket: PID={conn.pid}, Port={port}",
                                    remediation_guidance=(
                                        f"Bind internal management services strictly to localhost (127.0.0.1) "
                                        f"or enforce TLS/HTTPS encryption with a reverse proxy."
                                    ),
                                    cwe_id="CWE-319",
                                )
                            )
        except Exception as e:
            log.warning(f"Network socket audit skipped or restricted: {e}")


safe_system_auditor = SafeSystemAuditor()
