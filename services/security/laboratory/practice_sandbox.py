"""
JARVIS AI — In-Memory Vulnerable Practice Sandbox
=================================================
Self-contained, intentionally vulnerable mock environments for
hands-on defensive learning and mitigation verification.
Operates strictly in memory without interacting with production systems.
Targets:
- MockSQLTarget (Demonstrating string concatenation vs parameterized defenses)
- MockTemplateTarget (Demonstrating raw HTML injection vs contextual escaping)
- MockPathTarget (Demonstrating path traversal vs canonical sandbox validation)
"""

import html
import os
from typing import Dict, Any, List, Optional
from core.logger import log


class MockSQLTarget:
    """In-memory mock database demonstrating SQL injection and parameterized remediation."""

    def __init__(self):
        self._mock_users = [
            {"id": 1, "username": "admin", "role": "administrator", "secret": "token_admin_999"},
            {"id": 2, "username": "harsh", "role": "operator", "secret": "token_user_123"},
            {"id": 3, "username": "guest", "role": "guest", "secret": "token_guest_000"},
        ]

    def vulnerable_query(self, username_input: str) -> List[Dict[str, Any]]:
        """
        Simulates unparameterized SQL concatenation:
        SELECT * FROM users WHERE username = '{username_input}'
        """
        raw_sql = f"SELECT * FROM users WHERE username = '{username_input}'"
        log.info(f"MockSQLTarget executing raw concatenated query: {raw_sql}")

        # Simulate classic ' OR '1'='1 SQL injection
        if "' or '1'='1" in username_input.lower() or "' or 1=1" in username_input.lower():
            return list(self._mock_users)

        # Standard match
        return [u for u in self._mock_users if u["username"] == username_input]

    def secure_parameterized_query(self, username_input: str) -> List[Dict[str, Any]]:
        """
        Simulates parameterized query defense:
        SELECT * FROM users WHERE username = ? with (username_input,)
        """
        log.info("MockSQLTarget executing secure parameterized query.")
        # Exact literal match on username parameter — SQL injection is neutralized
        return [u for u in self._mock_users if u["username"] == username_input]


class MockTemplateTarget:
    """In-memory mock web template demonstrating XSS and contextual output encoding."""

    def vulnerable_render(self, user_comment: str) -> str:
        """Simulates raw, unescaped HTML string interpolation (vulnerable to XSS)."""
        return f"<div class='comment'><p>{user_comment}</p></div>"

    def secure_render(self, user_comment: str) -> str:
        """Applies HTML entity encoding, neutralizing script tag execution."""
        safe_comment = html.escape(user_comment, quote=True)
        return f"<div class='comment'><p>{safe_comment}</p></div>"


class MockPathTarget:
    """In-memory mock filesystem demonstrating path traversal and canonical sandbox defense."""

    def __init__(self, sandbox_root: str = "/var/app/storage"):
        self.sandbox_root = os.path.realpath(os.path.abspath(sandbox_root))

    def vulnerable_resolve(self, requested_filename: str) -> str:
        """Simulates naive path concatenation without traversal checks."""
        return os.path.normpath(os.path.join(self.sandbox_root, requested_filename))

    def secure_resolve(self, requested_filename: str) -> str:
        """
        Applies canonicalization and boundary verification,
        raising PermissionError on traversal attempts.
        """
        if "\0" in requested_filename:
            raise ValueError("Path traversal attempt: null byte detected.")

        candidate = os.path.realpath(os.path.abspath(os.path.join(self.sandbox_root, requested_filename)))
        try:
            if os.path.commonpath([self.sandbox_root, candidate]) != self.sandbox_root:
                raise PermissionError(f"Directory traversal blocked: '{requested_filename}' outside sandbox.")
        except ValueError:
            raise PermissionError(f"Directory traversal blocked: cross-drive escape.")

        return candidate


# Singletons for practice sandbox
mock_sql_target = MockSQLTarget()
mock_template_target = MockTemplateTarget()
mock_path_target = MockPathTarget()
