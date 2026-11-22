"""
JARVIS AI — Cybersecurity Laboratory Report Generator
======================================================
Renders standardized defensive security assessment and quarantine inspection reports
with executive scorecards, CVSS severity breakdowns, and actionable remediation guides.
"""

from services.security.laboratory.models import (
    LaboratoryAuditReport,
    FileAnalysisReport,
    FindingSeverity,
)


class SecurityReportGenerator:
    """Formats assessment metrics into structured markdown and executive briefings."""

    @staticmethod
    def format_audit_report(report: LaboratoryAuditReport) -> str:
        """Render a full system security audit report in markdown."""
        lines = [
            f"# 🛡️ Defensive Security Audit Report: {report.target_system}",
            f"**Assessment ID:** `{report.assessment_id}`  ",
            f"**Timestamp:** {report.conducted_at.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"**Audit Duration:** {report.duration_sec:.2f} seconds  ",
            f"**Overall Defensive Posture Score:** **{report.defensive_scorecard:.1f} / 100.0**  ",
            "",
            "## 📊 Executive Finding Summary",
            f"- **Total Identified Weaknesses:** {report.total_findings}",
            f"- 🔴 **Critical:** {report.severity_counts.get('CRITICAL', 0)}",
            f"- 🟠 **High:** {report.severity_counts.get('HIGH', 0)}",
            f"- 🟡 **Medium:** {report.severity_counts.get('MEDIUM', 0)}",
            f"- 🔵 **Low:** {report.severity_counts.get('LOW', 0)}",
            f"- ℹ️ **Info:** {report.severity_counts.get('INFO', 0)}",
            "",
            f"**Summary:** {report.executive_summary}",
            "",
            "---",
            "## 🔍 Detailed Security Findings & Remediation Plans",
        ]

        if not report.findings:
            lines.append("✅ **No immediate security weaknesses detected within audited scope.**")
        else:
            for idx, finding in enumerate(report.findings, start=1):
                sev_icon = {
                    FindingSeverity.CRITICAL: "🔴",
                    FindingSeverity.HIGH: "🟠",
                    FindingSeverity.MEDIUM: "🟡",
                    FindingSeverity.LOW: "🔵",
                    FindingSeverity.INFO: "ℹ️",
                }.get(finding.severity, "⚠️")

                lines.extend([
                    f"### {idx}. {sev_icon} [{finding.severity.value}] {finding.title}",
                    f"- **Finding ID:** `{finding.finding_id}`",
                    f"- **Category:** {finding.category} (CWE: `{finding.cwe_id or 'N/A'}`)",
                    f"- **CVSS v3.1 Score:** **{finding.cvss_score}**",
                    f"- **Affected Target:** `{finding.affected_target}`",
                    f"- **Description:** {finding.description}",
                    f"- **Evidence:** `{finding.evidence}`",
                    "",
                    f"**🛡️ Remediation Guidance:**",
                    f"> {finding.remediation_guidance}",
                    "",
                ])

        return "\n".join(lines)

    @staticmethod
    def format_file_analysis_report(report: FileAnalysisReport) -> str:
        """Render an isolated static quarantine file analysis report in markdown."""
        risk_icon = {
            "BENIGN": "🟢",
            "SUSPICIOUS": "🟡",
            "ANOMALOUS": "🟠",
            "HIGH_RISK": "🔴",
        }.get(report.risk_assessment, "⚠️")

        lines = [
            f"# 🔬 Quarantine File Analysis Report: {report.file_path}",
            f"**Risk Level:** {risk_icon} **{report.risk_assessment}**  ",
            f"**File Size:** {report.file_size_bytes:,} bytes  ",
            f"**Shannon Entropy:** **{report.shannon_entropy:.2f} / 8.0**  ",
            f"**Detected Type (Magic):** `{report.detected_mime_type}`  ",
            f"**Declared Extension:** `{report.declared_extension}`  ",
            f"**Executable Masquerading:** {'⚠️ YES' if report.is_masquerading else '✅ No'}",
            "",
            "## 🔑 Cryptographic Hashes",
            f"- **SHA-256:** `{report.sha256_hash}`",
            f"- **SHA-1:** `{report.sha1_hash}`",
            f"- **MD5:** `{report.md5_hash}`",
            "",
            "## ⚠️ Suspicious Indicators",
        ]

        if not report.suspicious_indicators:
            lines.append("✅ No anomalous structural indicators or executable masquerading detected.")
        else:
            for ind in report.suspicious_indicators:
                lines.append(f"- ⚠️ {ind}")

        lines.extend([
            "",
            "## 🛡️ Defensive Recommendations",
        ])
        for rec in report.defensive_recommendations:
            lines.append(f"• {rec}")

        return "\n".join(lines)


report_generator = SecurityReportGenerator()
