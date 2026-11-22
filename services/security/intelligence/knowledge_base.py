"""
JARVIS AI — Defensive Cybersecurity Knowledge Catalog
======================================================
Exhaustive, 100% offline security knowledge base covering all 10 core domains:
1. Computer Networking & OSI Model
2. TCP/IP, DNS, HTTP, HTTPS & TLS 1.3
3. Linux & Windows System Administration Hardening
4. Applied Cryptography, Ciphers & PKI
5. Web Application Security & OWASP Top 10
6. Authentication, Authorization & Zero Trust
7. Vulnerability Analysis & CWE/CVSS Taxonomies
8. Security Monitoring, SIEM, IDS/IPS & Telemetry
9. Defensive Security & Incident Response (NIST/SANS)
10. Secure Software Development & Threat Modeling (STRIDE)
"""

from typing import List, Dict, Optional
from services.security.intelligence.models import (
    SecurityTopic,
    SecurityDomain,
    VulnerabilityDef,
    SeverityLevel,
    HardeningChecklist,
    HardeningStep,
)

# ── 1. Comprehensive Domain Topics Catalog ────────────────────────────────────

KNOWLEDGE_TOPICS: List[SecurityTopic] = [
    # 1. Computer Networking
    SecurityTopic(
        topic_id="net-001",
        title="OSI Model & Network Layer Defense",
        domain=SecurityDomain.NETWORKING,
        summary="7-layer conceptual architecture for network communication and layer-specific defensive controls.",
        technical_details=(
            "The OSI model partitions network communications into 7 layers: "
            "1: Physical (cables, RF); 2: Data Link (MAC, ARP, 802.1Q VLAN tagging); "
            "3: Network (IP, ICMP, BGP routing); 4: Transport (TCP, UDP, SCTP); "
            "5: Session (RPC, NetBIOS); 6: Presentation (TLS, data encoding); "
            "7: Application (HTTP, DNS, SSH, SMTP). "
            "Defensive controls must be implemented across all layers: 802.1X port security (L2), "
            "firewall packet filtering and CIDR segmentation (L3), stateful connection tracking (L4), "
            "and application-layer inspection / WAF (L7)."
        ),
        threat_model="L2 ARP spoofing, MAC flooding, IP spoofing, routing table poison, and DDoS saturation.",
        defensive_remediations=[
            "Enable Dynamic ARP Inspection (DAI) and DHCP Snooping on managed switches.",
            "Enforce strict ingress and egress IP filtering (BCP 38 / RFC 2827) to block spoofed source IPs.",
            "Deploy network micro-segmentation using VLANs and zero-trust perimeter firewalls.",
        ],
        tags=["osi", "networking", "ethernet", "vlan", "arp", "defense_in_depth"],
        rfc_or_standards=["RFC 791", "RFC 826", "IEEE 802.1Q"],
    ),

    # 2. TCP/IP, DNS, HTTP/S & TLS 1.3
    SecurityTopic(
        topic_id="proto-001",
        title="TCP 3-Way Handshake, SYN Flood Mitigation & State Tracking",
        domain=SecurityDomain.PROTOCOLS,
        summary="Connection establishment mechanics, TCP state exhaustion attacks, and kernel SYN cookie defense.",
        technical_details=(
            "TCP connections establish via a 3-way handshake: Client sends SYN (seq=X); "
            "Server responds with SYN-ACK (seq=Y, ack=X+1) and allocates a Transmission Control Block (TCB); "
            "Client completes with ACK (ack=Y+1). "
            "Under a SYN flood attack, attackers send spoofed SYN packets without sending final ACKs, "
            "filling the server's SYN backlog queue and denying legitimate traffic. "
            "SYN cookies solve this by encoding TCB parameters into the initial sequence number Y = hash(secrets, IPs, ports, MSS) "
            "without allocating memory until the client's ACK is verified."
        ),
        threat_model="Denial of Service (DoS/DDoS) via SYN backlog queue resource exhaustion.",
        defensive_remediations=[
            "Enable SYN cookies: sysctl -w net.ipv4.tcp_syncookies=1 on Linux hosts.",
            "Reduce SYN-ACK retry counts: sysctl -w net.ipv4.tcp_synack_retries=2.",
            "Deploy stateful firewall rate limiting or upstream DDoS scrubbing mitigation.",
        ],
        tags=["tcp", "syn_flood", "ddos", "networking", "kernel_tuning"],
        rfc_or_standards=["RFC 793", "RFC 4987"],
    ),
    SecurityTopic(
        topic_id="proto-002",
        title="DNS Security, Cache Poisoning & DNSSEC Validation",
        domain=SecurityDomain.PROTOCOLS,
        summary="Domain Name System resolution vulnerabilities, Kaminsky cache poisoning, and DNSSEC cryptographic validation.",
        technical_details=(
            "Standard DNS operates over UDP port 53 in cleartext. In Dan Kaminsky's cache poisoning attack, "
            "an attacker floods a recursive resolver with forged responses matching guessed 16-bit Transaction IDs (TXIDs) "
            "and UDP source ports before the authoritative server replies, hijacking resolution for a domain. "
            "DNSSEC (DNS Security Extensions) mitigates this by applying digital signatures (RRSIG) "
            "over resource record sets (RRsets) validated through a hierarchical chain of trust up to the Root Trust Anchor (DS records)."
        ),
        threat_model="Man-in-the-Middle (MitM) resolution hijacking, phishing redirect, and BGP routing diversion.",
        defensive_remediations=[
            "Enable UDP Source Port Randomization (0x20 encoding / ephemeral range 1024-65535).",
            "Enable strict DNSSEC validation on recursive resolvers (e.g. Unbound, BIND).",
            "Implement DNS over HTTPS (DoH) or DNS over TLS (DoT) to protect last-mile privacy.",
        ],
        tags=["dns", "dnssec", "cache_poisoning", "doh", "dot"],
        rfc_or_standards=["RFC 4033", "RFC 4034", "RFC 4035", "RFC 5452"],
    ),
    SecurityTopic(
        topic_id="proto-003",
        title="TLS 1.3 Architecture, Forward Secrecy & Secure Cipher Suites",
        domain=SecurityDomain.PROTOCOLS,
        summary="Transport Layer Security 1.3 protocol improvements, 1-RTT handshake, and deprecation of legacy ciphers.",
        technical_details=(
            "TLS 1.3 (RFC 8446) redesigns the secure handshake: reduced from 2-RTT to 1-RTT by combining key share "
            "exchange (ECDHE) into the ClientHello. It completely removes obsolete, insecure primitives: "
            "RSA key transport, CBC-mode ciphers, static Diffie-Hellman, RC4, MD5, and SHA-1. "
            "Only Authenticated Encryption with Associated Data (AEAD) algorithms are allowed: "
            "TLS_AES_256_GCM_SHA384, TLS_CHACHA20_POLY1305_SHA256, and TLS_AES_128_GCM_SHA256. "
            "Ephemeral Diffie-Hellman (ECDHE with X25519 or P-256) guarantees Perfect Forward Secrecy (PFS), "
            "ensuring past sessions cannot be decrypted even if the server's long-term private key is later compromised."
        ),
        threat_model="Eavesdropping, session decryption via compromised private keys, and protocol downgrade attacks.",
        defensive_remediations=[
            "Enforce TLS 1.3 and TLS 1.2 minimum; permanently disable SSLv3, TLS 1.0, and TLS 1.1.",
            "Deploy HTTP Strict Transport Security (HSTS) with max-age=31536000 and includeSubDomains.",
            "Configure Certificate Transparency (CT) logging and OCSP Stapling.",
        ],
        tags=["tls", "tls13", "https", "cryptography", "pfs", "aead", "hsts"],
        rfc_or_standards=["RFC 8446", "RFC 6797"],
    ),

    # 3. Linux & Windows System Administration
    SecurityTopic(
        topic_id="sys-001",
        title="Linux Privilege Separation, SUID Hazards & Mandatory Access Control",
        domain=SecurityDomain.SYSADMIN,
        summary="POSIX permission boundaries, SUID binary audit, and SELinux / AppArmor policy confinement.",
        technical_details=(
            "Linux systems rely on Discretionary Access Control (DAC) via user (UID), group (GID), and permissions (rwx). "
            "Binaries with the Set-UID (SUID, mode 4000) or SGID (mode 2000) bit execute with the privileges of the file owner (typically root). "
            "If an SUID binary contains command injection or vulnerable environment variable parsing (e.g. PATH injection), "
            "unprivileged users can escalate to root. "
            "Mandatory Access Control (MAC) frameworks (SELinux type enforcement, AppArmor profiles) enforce confinement: "
            "even if a root process is compromised, its actions are restricted to explicitly permitted domain transitions."
        ),
        threat_model="Local Privilege Escalation (LPE) from unprivileged local user to root.",
        defensive_remediations=[
            "Audit all SUID binaries regularly: find / -perm -4000 -type f 2>/dev/null.",
            "Mount temporary and user-writable partitions with 'nosuid,nodev,noexec' options.",
            "Keep SELinux in Enforcing mode: getenforce -> Enforcing.",
        ],
        tags=["linux", "suid", "privilege_escalation", "selinux", "apparmor", "hardening"],
        rfc_or_standards=["NIST SP 800-123", "CIS Linux Benchmarks"],
    ),
    SecurityTopic(
        topic_id="sys-002",
        title="Windows Security Architecture: SIDs, DACLs, SACLs & UAC",
        domain=SecurityDomain.SYSADMIN,
        summary="Windows NT object access tokens, discretionary access control lists, and User Account Control integrity levels.",
        technical_details=(
            "Windows secures securable objects (files, processes, registry keys) using Security Descriptors containing: "
            "1. Owner SID (Security Identifier); 2. DACL (Discretionary Access Control List) with Access Control Entries (ACEs); "
            "3. SACL (System Access Control List) defining auditing rules. "
            "User Account Control (UAC) enforces integrity levels (Untrusted, Low, Medium, High, System). "
            "Interactive administrator accounts run standard processes at Medium Integrity; elevated privileges "
            "require token duplication and affirmative consent via the secure desktop."
        ),
        threat_model="Token manipulation, DLL hijacking, unquoted service path exploitation, and privilege escalation.",
        defensive_remediations=[
            "Configure UAC to 'Always Notify' to enforce secure desktop consent.",
            "Restrict access to sensitive registry keys and services (services.msc DACL audits).",
            "Enable Credential Guard (Virtualization-Based Security) to protect LSASS memory against Mimikatz credential dumping.",
        ],
        tags=["windows", "uac", "dacl", "sids", "lsass", "credential_guard"],
        rfc_or_standards=["Microsoft Security Baselines", "CIS Windows Benchmarks"],
    ),

    # 4. Modern Cryptography & PKI
    SecurityTopic(
        topic_id="crypto-001",
        title="Symmetric vs Asymmetric Cryptography & Key Management",
        domain=SecurityDomain.CRYPTOGRAPHY,
        summary="Mathematical distinctions between symmetric block/stream ciphers, asymmetric trapdoor functions, and key lifecycles.",
        technical_details=(
            "Symmetric ciphers (AES, ChaCha20) use a single shared secret key for encryption and decryption. "
            "AES operates on 128-bit blocks using substitution-permutation networks; ChaCha20 is a 256-bit stream cipher using ARX operations. "
            "Asymmetric algorithms (RSA, ECDSA, Ed25519, X25519) use mathematically linked key pairs (public/private). "
            "RSA relies on prime factorization intractability; Elliptic Curve Cryptography relies on the Discrete Logarithm Problem "
            "over cyclic groups, achieving equivalent 128-bit security with 256-bit keys (compared to 3072-bit keys in RSA). "
            "Hybrid cryptosystems combine both: asymmetric key exchange (ECDH) securely negotiates an ephemeral symmetric session key."
        ),
        threat_model="Key leakage, brute-force exhaustion, mathematical cryptanalysis, and side-channel timing attacks.",
        defensive_remediations=[
            "Use minimum 256-bit symmetric keys (AES-256) and 256-bit elliptic curves (X25519 / NIST P-256).",
            "Always use constant-time comparison algorithms (e.g. hmac.compare_digest) for verification.",
            "Employ hardware security modules (HSM) or TPM 2.0 chips for master root key storage.",
        ],
        tags=["cryptography", "aes", "ecc", "rsa", "key_management", "tpm"],
        rfc_or_standards=["NIST SP 800-57", "FIPS 140-3", "RFC 7748"],
    ),

    # 5. Web Application Security & OWASP Top 10
    SecurityTopic(
        topic_id="web-001",
        title="OWASP Top 10 Web Application Defensive Architecture",
        domain=SecurityDomain.WEB_SECURITY,
        summary="The 10 most critical web application security risks and comprehensive defensive coding countermeasures.",
        technical_details=(
            "OWASP Top 10 (2021) categorizes the most severe web flaws: "
            "A01: Broken Access Control (improper authorization, IDOR); "
            "A02: Cryptographic Failures (hardcoded keys, cleartext transmission); "
            "A03: Injection (SQL, Command, LDAP); "
            "A04: Insecure Design (lack of threat modeling); "
            "A05: Security Misconfiguration (default passwords, unneeded features); "
            "A06: Vulnerable and Outdated Components; "
            "A07: Identification and Authentication Failures (brute force, weak passwords); "
            "A08: Software and Data Integrity Failures (untrusted CI/CD, insecure deserialization); "
            "A09: Security Logging and Monitoring Failures; "
            "A10: Server-Side Request Forgery (SSRF). "
            "Mitigations require parameterization, strict input validation, contextual output encoding, and defense-in-depth."
        ),
        threat_model="Remote Code Execution (RCE), database breach, credential theft, and full server takeover.",
        defensive_remediations=[
            "Enforce parameterized SQL queries and Object-Relational Mappers (ORMs); never concatenate user inputs.",
            "Implement a strict Content Security Policy (CSP): default-src 'self'.",
            "Validate all access control checks server-side on every request, not client-side.",
        ],
        tags=["owasp", "web_security", "injection", "access_control", "ssrf", "xss"],
        rfc_or_standards=["OWASP Top 10:2021", "CWE/SANS Top 25"],
    ),

    # 6. Authentication, Authorization & Zero Trust
    SecurityTopic(
        topic_id="auth-001",
        title="Zero Trust Architecture & Modern Multi-Factor Protocols (FIDO2 / WebAuthn)",
        domain=SecurityDomain.AUTHENTICATION,
        summary="Eliminating implicit trust perimeters, passwordless WebAuthn public-key authentication, and continuous verification.",
        technical_details=(
            "Traditional perimeter defense ('castle-and-moat') grants broad access once inside the internal LAN. "
            "Zero Trust Architecture (NIST SP 800-207) enforces: 'Never Trust, Always Verify'. "
            "Every access request is authenticated, authorized, and encrypted regardless of network location. "
            "FIDO2 / WebAuthn provides phishing-resistant authentication using hardware-backed asymmetric key pairs: "
            "the private key never leaves the authenticator (YubiKey, TPM, biometric sensor), "
            "and authentication challenges are cryptographically bound to the web origin (RP ID), eliminating credential phishing."
        ),
        threat_model="Phishing, credential stuffing, adversary-in-the-middle (AiTM) proxy attacks, and lateral movement.",
        defensive_remediations=[
            "Deprecate SMS and push-notification OTPs in favor of FIDO2 / WebAuthn passkeys or hardware tokens.",
            "Implement Micro-Segmentation: isolate workloads with strict identity-aware proxy policies.",
            "Enforce Continuous Adaptive Risk and Trust Assessment (CARTA) monitoring device health and telemetry.",
        ],
        tags=["zero_trust", "fido2", "webauthn", "mfa", "nist", "identity"],
        rfc_or_standards=["NIST SP 800-207", "W3C WebAuthn Level 2"],
    ),

    # 7. Vulnerability Analysis & Scoring
    SecurityTopic(
        topic_id="vuln-001",
        title="CVSS v3.1 Metrics Calculation & Vulnerability Lifecycle Management",
        domain=SecurityDomain.VULNERABILITY,
        summary="Common Vulnerability Scoring System metrics, Base Score equations, and remediation prioritization.",
        technical_details=(
            "CVSS v3.1 calculates severity (0.0 to 10.0) across three metric groups: "
            "1. Base Metrics: Attack Vector (AV: Network, Adjacent, Local, Physical); "
            "Attack Complexity (AC: Low, High); Privileges Required (PR: None, Low, High); "
            "User Interaction (UI: None, Required); Scope (S: Unchanged, Changed); "
            "Confidentiality, Integrity, Availability Impacts (C/I/A: None, Low, High). "
            "2. Temporal Metrics: Exploit Code Maturity (E), Remediation Level (RL), Report Confidence (RC). "
            "3. Environmental Metrics: Modified Base Metrics tailored to organizational context. "
            "Scores 0.1-3.9 are Low, 4.0-6.9 Medium, 7.0-8.9 High, and 9.0-10.0 Critical."
        ),
        threat_model="Unpatched known vulnerabilities (1-day exploits) targeting exposed public services.",
        defensive_remediations=[
            "Automate software bill of materials (SBOM) scanning across all application dependencies.",
            "Establish Service Level Agreements (SLAs) for patching: Critical (≤ 72h), High (≤ 14d).",
            "Prioritize remediation using the Known Exploited Vulnerabilities (KEV) catalog from CISA.",
        ],
        tags=["cvss", "vulnerability_management", "cisa_kev", "patching", "sbom"],
        rfc_or_standards=["FIRST CVSS v3.1 Specification", "NIST SP 800-40"],
    ),

    # 8. Security Monitoring & SIEM
    SecurityTopic(
        topic_id="mon-001",
        title="SIEM Architecture, Log Telemetry & Intrusion Detection (IDS/IPS)",
        domain=SecurityDomain.MONITORING,
        summary="Centralized security monitoring, Sysmon event collection, Sigma detection rules, and anomaly detection.",
        technical_details=(
            "Security Information and Event Management (SIEM) aggregates telemetry from endpoints, network sensors, and firewalls. "
            "On Windows, Microsoft Sysmon logs critical telemetry: Event ID 1 (Process Creation with CLI args), "
            "Event ID 3 (Network Connection), Event ID 7 (Image Loaded/DLL Injection), Event ID 10 (ProcessAccess/LSASS dumping). "
            "Intrusion Detection Systems (Suricata, Snort) analyze packet payloads against signature rules (Snort rules) "
            "and protocol anomalies (Zeek). Detection engineering uses open standards like Sigma rules to standardize alerts."
        ),
        threat_model="Undetected advanced persistent threats (APTs), living-off-the-land binaries (LOLBins), and data exfiltration.",
        defensive_remediations=[
            "Deploy Sysmon with a hardened configuration (e.g. SwiftOnSecurity / Olaf Hartong modular configs).",
            "Forward logs via TLS to an immutable, append-only centralized logging repository.",
            "Implement automated correlation alerts for abnormal parent-child process chains (e.g. WINWORD.EXE -> CMD.EXE).",
        ],
        tags=["siem", "sysmon", "ids", "ips", "sigma", "detection_engineering"],
        rfc_or_standards=["NIST SP 800-92", "MITRE ATT&CK Framework"],
    ),

    # 9. Defensive Security & Incident Response
    SecurityTopic(
        topic_id="ir-001",
        title="NIST SP 800-61 Incident Handling Lifecycle & SANS PICERL",
        domain=SecurityDomain.DEFENSIVE_OPS,
        summary="Standardized methodologies for preparing, detecting, containing, and recovering from cybersecurity incidents.",
        technical_details=(
            "NIST SP 800-61 Rev. 2 outlines four phases of Computer Security Incident Handling: "
            "1. Preparation: tools, processes, trained personnel, hardened systems; "
            "2. Detection & Analysis: monitoring indicators of compromise (IOCs), triage, scoping severity; "
            "3. Containment, Eradication & Recovery: short-term containment (network isolation, token revocation), "
            "long-term containment, eliminating malware/backdoors, restoring clean systems from validated backups; "
            "4. Post-Incident Activity: Lessons learned, updating playbooks, evidence retention for chain of custody. "
            "SANS defines the PICERL mnemonic: Preparation, Identification, Containment, Eradication, Recovery, Lessons Learned."
        ),
        threat_model="Ransomware propagation, credential compromise, active unauthorized persistence, and data extortion.",
        defensive_remediations=[
            "Maintain immutable, air-gapped backups tested regularly with automated recovery exercises.",
            "Formulate pre-authorized containment playbooks (e.g. automated endpoint network isolation).",
            "Conduct quarterly tabletop incident simulation exercises.",
        ],
        tags=["incident_response", "nist", "sans", "picerl", "containment", "playbooks"],
        rfc_or_standards=["NIST SP 800-61 Rev. 2", "ISO/IEC 27035"],
    ),

    # 10. Secure Software Development (SSDLC) & Threat Modeling
    SecurityTopic(
        topic_id="dev-001",
        title="Threat Modeling (STRIDE) & Secure Development Lifecycle (SSDLC)",
        domain=SecurityDomain.SECURE_DEV,
        summary="Identifying architectural weaknesses early using STRIDE and embedding automated security gates in CI/CD pipelines.",
        technical_details=(
            "The STRIDE threat model categorizes architectural security risks: "
            "S - Spoofing Identity (authenticity violation); "
            "T - Tampering with Data (integrity violation); "
            "R - Repudiation (non-repudiation violation); "
            "I - Information Disclosure (confidentiality violation); "
            "D - Denial of Service (availability violation); "
            "E - Elevation of Privilege (authorization violation). "
            "In a Secure SDLC, security controls shift left: Threat Modeling during architecture design, "
            "Static Application Security Testing (SAST) and Software Composition Analysis (SCA) during build, "
            "Dynamic Application Security Testing (DAST) in staging, and runtime monitoring (RASP/WAF) in production."
        ),
        threat_model="Architectural security debt, vulnerable third-party dependencies, and supply chain poisoning.",
        defensive_remediations=[
            "Perform STRIDE threat modeling on all new system components and data flows before implementation.",
            "Integrate automated SAST and dependency linters into git pre-commit hooks and CI/CD pipelines.",
            "Enforce branch protection rules with mandatory peer reviews and signed cryptographic git commits.",
        ],
        tags=["stride", "threat_modeling", "ssdlc", "sast", "dast", "supply_chain"],
        rfc_or_standards=["NIST SP 800-218 (SSDF)", "OWASP SAMM"],
    ),
]

# ── 2. CWE Vulnerability Definitions Catalog ──────────────────────────────────

VULNERABILITY_CATALOG: List[VulnerabilityDef] = [
    VulnerabilityDef(
        cwe_id="CWE-89",
        name="Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')",
        owasp_category="A03:2021-Injection",
        cvss_base_score=9.8,
        severity=SeverityLevel.CRITICAL,
        description="The software constructs all or part of an SQL command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended SQL command.",
        attack_vector="Network (AV:N)",
        impact="Full unauthorized read/write access to database records, authentication bypass, data destruction, and potential OS command execution via xp_cmdshell.",
        mitigation_guidance="Use parameterized queries / prepared statements exclusively. Never concatenate strings into SQL statements. Enforce least-privilege database user accounts.",
        secure_code_example=(
            "# VULNERABLE: cursor.execute(f'SELECT * FROM users WHERE name = \"{user_input}\"')\n"
            "# SECURE (Parameterized):\n"
            "cursor.execute('SELECT * FROM users WHERE name = ?', (user_input,))"
        ),
        references=["https://cwe.mitre.org/data/definitions/89.html", "OWASP SQL Injection Prevention Cheat Sheet"],
    ),
    VulnerabilityDef(
        cwe_id="CWE-79",
        name="Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')",
        owasp_category="A03:2021-Injection",
        cvss_base_score=7.3,
        severity=SeverityLevel.HIGH,
        description="The software does not neutralize or incorrectly neutralizes user-controllable input before it is placed in output that is used as a web page that is served to other users.",
        attack_vector="Network (AV:N)",
        impact="Execution of malicious JavaScript in victim's browser, session hijacking via stolen cookies/tokens, keystroke logging, and page defacement.",
        mitigation_guidance="Contextual output encoding (HTML, JavaScript, CSS context), enable Content Security Policy (CSP), mark sensitive cookies with HttpOnly and SameSite=Strict.",
        secure_code_example=(
            "// SECURE: element.textContent = userSuppliedString; // Prevents HTML parsing\n"
            "// Header: Content-Security-Policy: default-src 'self';"
        ),
        references=["https://cwe.mitre.org/data/definitions/79.html", "OWASP XSS Prevention Cheat Sheet"],
    ),
    VulnerabilityDef(
        cwe_id="CWE-22",
        name="Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')",
        owasp_category="A01:2021-Broken Access Control",
        cvss_base_score=7.5,
        severity=SeverityLevel.HIGH,
        description="The software uses external input to construct a pathname that should be within a restricted directory, but it does not properly neutralize sequences such as '..' that can resolve to a location that is outside of that directory.",
        attack_vector="Network (AV:N)",
        impact="Unauthorized reading or overwriting of sensitive configuration files, password hashes, source code, or critical operating system binaries.",
        mitigation_guidance="Canonicalize paths using realpath(), reject input containing '..' or null bytes, verify canonical path resides within whitelisted root directory with os.path.commonpath.",
        secure_code_example=(
            "canonical = os.path.realpath(os.path.abspath(user_path))\n"
            "if os.path.commonpath([SANDBOX_ROOT, canonical]) != SANDBOX_ROOT:\n"
            "    raise PermissionError('Path traversal detected.')"
        ),
        references=["https://cwe.mitre.org/data/definitions/22.html"],
    ),
    VulnerabilityDef(
        cwe_id="CWE-78",
        name="Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')",
        owasp_category="A03:2021-Injection",
        cvss_base_score=9.8,
        severity=SeverityLevel.CRITICAL,
        description="The software constructs all or part of an OS command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended OS command.",
        attack_vector="Network (AV:N)",
        impact="Complete compromise of host operating system with privileges of the executing application process.",
        mitigation_guidance="Avoid invoking OS shell interpreters (never use shell=True in subprocess). Pass arguments as a discrete array of strings. Enforce strict parameter whitelisting.",
        secure_code_example=(
            "# VULNERABLE: subprocess.run(f'ping -c 1 {host}', shell=True)\n"
            "# SECURE: subprocess.run(['ping', '-c', '1', validated_ip], shell=False)"
        ),
        references=["https://cwe.mitre.org/data/definitions/78.html"],
    ),
    VulnerabilityDef(
        cwe_id="CWE-352",
        name="Cross-Site Request Forgery (CSRF)",
        owasp_category="A01:2021-Broken Access Control",
        cvss_base_score=8.8,
        severity=SeverityLevel.HIGH,
        description="The web application does not verify that a request was intentionally generated by the user who submitted it, allowing malicious websites to trick the victim's browser into executing unwanted actions.",
        attack_vector="Network (AV:N)",
        impact="Unintended state-changing requests executed on victim's behalf: email change, password reset, funds transfer.",
        mitigation_guidance="Deploy cryptographically random anti-CSRF tokens verified per request (Synchronizer Token Pattern) and set SameSite=Lax or SameSite=Strict on session cookies.",
        secure_code_example=(
            "Set-Cookie: session_id=xyz; Secure; HttpOnly; SameSite=Strict;\n"
            "X-CSRF-Token: cryptographically_random_hmac_token"
        ),
        references=["https://cwe.mitre.org/data/definitions/352.html"],
    ),
    VulnerabilityDef(
        cwe_id="CWE-918",
        name="Server-Side Request Forgery (SSRF)",
        owasp_category="A10:2021-Server-Side Request Forgery",
        cvss_base_score=8.6,
        severity=SeverityLevel.HIGH,
        description="The web server receives a URL or network address from an upstream component and retrieves data from it without validating that the destination is safe, allowing attackers to access internal private networks.",
        attack_vector="Network (AV:N)",
        impact="Access to cloud metadata endpoints (169.254.169.254), internal microservices, local databases, and port scanning of internal network.",
        mitigation_guidance="Enforce strict URL allowlists. Parse URL, resolve IP, and reject private/reserved ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 127.0.0.0/8, 169.254.0.0/16).",
        secure_code_example=(
            "ip = socket.gethostbyname(parsed_domain)\n"
            "if ipaddress.ip_address(ip).is_private or ipaddress.ip_address(ip).is_loopback:\n"
            "    raise ValueError('SSRF blocked: private network destination.')"
        ),
        references=["https://cwe.mitre.org/data/definitions/918.html", "OWASP SSRF Cheat Sheet"],
    ),
    VulnerabilityDef(
        cwe_id="CWE-502",
        name="Deserialization of Untrusted Data",
        owasp_category="A08:2021-Software and Data Integrity Failures",
        cvss_base_score=9.8,
        severity=SeverityLevel.CRITICAL,
        description="The application deserializes untrusted data without sufficiently verifying that the resulting data will be valid, allowing attackers to execute arbitrary code via gadget chains.",
        attack_vector="Network (AV:N)",
        impact="Remote Code Execution (RCE) on the server processing the serialized object.",
        mitigation_guidance="Never deserialize untrusted data with formats that permit code execution (Python pickle, Java ObjectInputStream, YAML load). Use safe serialization like JSON.",
        secure_code_example=(
            "# VULNERABLE: data = pickle.loads(raw_user_bytes)\n"
            "# SECURE: data = json.loads(raw_user_bytes.decode('utf-8'))"
        ),
        references=["https://cwe.mitre.org/data/definitions/502.html"],
    ),
    VulnerabilityDef(
        cwe_id="CWE-798",
        name="Use of Hard-coded Credentials",
        owasp_category="A07:2021-Identification and Authentication Failures",
        cvss_base_score=7.5,
        severity=SeverityLevel.HIGH,
        description="The software contains hard-coded credentials, such as a password or cryptographic key, which can be extracted through reverse engineering or source code inspection.",
        attack_vector="Network / Local (AV:N)",
        impact="Complete bypass of authentication mechanisms by any actor who gains access to source code or binary.",
        mitigation_guidance="Inject credentials at runtime via environment variables or secret managers (HashiCorp Vault, AWS Secrets Manager). Enforce pre-commit git secret scanning.",
        secure_code_example=(
            "# VULNERABLE: API_KEY = 'secret123'\n"
            "# SECURE: API_KEY = os.environ.get('JARVIS_API_KEY')\n"
            "if not API_KEY: raise RuntimeError('Missing required secret in environment.')"
        ),
        references=["https://cwe.mitre.org/data/definitions/798.html"],
    ),
]

# ── 3. Standard Hardening Checklists ─────────────────────────────────────────

LINUX_HARDENING_CHECKLIST = HardeningChecklist(
    target_system="Linux Server (Ubuntu / Debian / RHEL)",
    description="Comprehensive defensive baseline for enterprise Linux server hardening.",
    steps=[
        HardeningStep(
            step_id="lnx-01",
            title="Disable Root SSH Login & Password Auth",
            category="SSH Hardening",
            description="Permit SSH access exclusively via cryptographic public keys and non-root user accounts.",
            verification_check="grep -E '^(PermitRootLogin|PasswordAuthentication)' /etc/ssh/sshd_config",
            recommended_config="PermitRootLogin no\nPasswordAuthentication no\nPubkeyAuthentication yes",
        ),
        HardeningStep(
            step_id="lnx-02",
            title="Enable and Configure UFW / iptables Firewall",
            category="Network Defense",
            description="Block all inbound traffic by default, permitting only explicitly required service ports.",
            verification_check="ufw status verbose",
            recommended_config="ufw default deny incoming\nufw default allow outgoing\nufw allow 22/tcp\nufw enable",
        ),
        HardeningStep(
            step_id="lnx-03",
            title="Kernel Network Parameter Hardening (sysctl)",
            category="Kernel Hardening",
            description="Prevent IP spoofing, drop ICMP redirects, and enable SYN cookies in sysctl.",
            verification_check="sysctl net.ipv4.tcp_syncookies net.ipv4.conf.all.rp_filter",
            recommended_config=(
                "net.ipv4.tcp_syncookies = 1\n"
                "net.ipv4.conf.all.rp_filter = 1\n"
                "net.ipv4.conf.all.accept_redirects = 0\n"
                "net.ipv4.conf.all.send_redirects = 0\n"
                "net.ipv4.icmp_echo_ignore_broadcasts = 1"
            ),
        ),
        HardeningStep(
            step_id="lnx-04",
            title="Enforce Automatic Security Updates",
            category="Patch Management",
            description="Ensure critical operating system security patches install automatically.",
            verification_check="systemctl is-active unattended-upgrades",
            recommended_config="apt-get install unattended-upgrades && dpkg-reconfigure -plow unattended-upgrades",
        ),
    ],
)
