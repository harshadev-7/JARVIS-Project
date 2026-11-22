"""
JARVIS AI — Isolated Static File Quarantine Analyzer
=====================================================
Performs safe, passive static analysis on suspicious files without execution.
Analyzes:
- Cryptographic hash fingerprints (SHA-256, SHA-1, MD5).
- Shannon Entropy calculation (detecting packed/encrypted/obfuscated sections).
- Magic byte header inspection (detecting executable masquerading).
- Static string and pattern triage (detecting hardcoded IPs, shell invocations, keys).
"""

import os
import math
import hashlib
import collections
from typing import List, Dict, Tuple, Optional

from services.security.laboratory.models import FileAnalysisReport
from core.logger import log

# Known magic byte signatures
MAGIC_SIGNATURES: List[Tuple[bytes, str, str]] = [
    (b"\x4d\x5a", "application/x-dosexec", "PE Executable (Windows EXE/DLL)"),
    (b"\x7f\x45\x4c\x46", "application/x-executable", "ELF Binary (Linux Executable)"),
    (b"\x50\x4b\x03\x04", "application/zip", "ZIP Archive"),
    (b"\x25\x50\x44\x46", "application/pdf", "PDF Document"),
    (b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a", "image/png", "PNG Image"),
    (b"\xff\xd8\xff", "image/jpeg", "JPEG Image"),
    (b"\x1f\x8b\x08", "application/gzip", "GZIP Compressed"),
    (b"\xca\xfe\xba\xbe", "application/java-vm", "Java Class File"),
]


class StaticFileAnalyzer:
    """Safe, isolated static file inspection engine (Zero Execution)."""

    @staticmethod
    def calculate_shannon_entropy(data: bytes) -> float:
        """
        Compute Shannon Entropy: H = -sum(p_i * log2(p_i)).
        Returns a float between 0.0 (uniform) and 8.0 (completely random / encrypted).
        """
        if not data:
            return 0.0

        length = len(data)
        frequencies = collections.Counter(data)
        entropy = 0.0

        for count in frequencies.values():
            p = count / length
            entropy -= p * math.log2(p)

        return round(entropy, 4)

    def analyze_file(self, file_path: str) -> FileAnalysisReport:
        """
        Perform passive static quarantine inspection on a file.
        NEVER executes the file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target file does not exist: {file_path}")

        canonical_path = os.path.realpath(os.path.abspath(file_path))
        file_size = os.path.getsize(canonical_path)
        ext = os.path.splitext(canonical_path)[1].lower()

        sha256_h = hashlib.sha256()
        sha1_h = hashlib.sha1()
        md5_h = hashlib.md5()

        # Read file safely in 64 KB chunks
        raw_header = b""
        data_sample = b""
        max_sample = 512 * 1024  # Sample up to 512 KB for entropy / strings

        with open(canonical_path, "rb") as f:
            raw_header = f.read(16)
            f.seek(0)
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                sha256_h.update(chunk)
                sha1_h.update(chunk)
                md5_h.update(chunk)
                if len(data_sample) < max_sample:
                    data_sample += chunk[: max_sample - len(data_sample)]

        entropy = self.calculate_shannon_entropy(data_sample)
        detected_mime, format_desc = self._detect_magic_type(raw_header)

        # Masquerading detection (e.g. PE binary masquerading as .txt, .png, or .pdf)
        is_masquerading = False
        suspicious_indicators = []

        if detected_mime in ("application/x-dosexec", "application/x-executable"):
            if ext in (".txt", ".pdf", ".png", ".jpg", ".jpeg", ".mp4", ".docx", ".xlsx"):
                is_masquerading = True
                suspicious_indicators.append(
                    f"Executable Masquerading: File declared as '{ext}' but contains '{format_desc}' header (MZ/ELF)."
                )

        if entropy > 7.2:
            suspicious_indicators.append(
                f"High Shannon Entropy ({entropy:.2f} / 8.0): File content is likely packed, obfuscated, or encrypted."
            )

        # String pattern extraction (passive)
        extracted_strings = self._extract_printable_strings(data_sample)
        found_patterns = self._scan_suspicious_patterns(extracted_strings)
        suspicious_indicators.extend(found_patterns)

        # Risk assessment
        risk_level = "BENIGN"
        recommendations = ["File matches standard structural baselines."]

        if is_masquerading or len(suspicious_indicators) >= 2:
            risk_level = "HIGH_RISK"
            recommendations = [
                "Quarantine file immediately: Do not execute or launch in standard user context.",
                "Review embedded network indicators and file origin source.",
                "Submit hash to internal threat intelligence database.",
            ]
        elif suspicious_indicators:
            risk_level = "SUSPICIOUS"
            recommendations = [
                "Exercise caution: File exhibits anomalous entropy or embedded indicators.",
                "Verify file integrity and authenticity using trusted vendor digital signatures.",
            ]

        return FileAnalysisReport(
            file_path=canonical_path,
            file_size_bytes=file_size,
            sha256_hash=sha256_h.hexdigest(),
            sha1_hash=sha1_h.hexdigest(),
            md5_hash=md5_h.hexdigest(),
            shannon_entropy=entropy,
            detected_mime_type=detected_mime,
            declared_extension=ext or ".unknown",
            is_masquerading=is_masquerading,
            suspicious_indicators=suspicious_indicators,
            risk_assessment=risk_level,
            defensive_recommendations=recommendations,
        )

    def _detect_magic_type(self, header: bytes) -> Tuple[str, str]:
        """Match file header bytes against known magic signatures."""
        for sig, mime, desc in MAGIC_SIGNATURES:
            if header.startswith(sig):
                return mime, desc
        return "application/octet-stream", "Generic Binary or Text"

    def _extract_printable_strings(self, data: bytes, min_len: int = 4) -> List[str]:
        """Extract continuous sequences of printable ASCII/UTF-8 characters."""
        result = []
        current = []
        for byte in data:
            if 32 <= byte <= 126:  # Printable ASCII
                current.append(chr(byte))
            else:
                if len(current) >= min_len:
                    result.append("".join(current))
                current = []
        if len(current) >= min_len:
            result.append("".join(current))
        return result

    def _scan_suspicious_patterns(self, strings: List[str]) -> List[str]:
        """Scan extracted strings for dangerous indicators."""
        indicators = []
        joined = "\n".join(strings).lower()

        if "powershell -enc" in joined or "powershell.exe -w hidden" in joined:
            indicators.append("Suspicious Command: Encoded or hidden PowerShell invocation detected.")
        if "cmd.exe /c" in joined:
            indicators.append("Command Line Spawn: Hardcoded cmd.exe invocation found.")
        if "begin rsa private key" in joined or "begin openssh private key" in joined:
            indicators.append("Cryptographic Leak: Embedded private key header discovered.")

        return indicators


static_file_analyzer = StaticFileAnalyzer()
