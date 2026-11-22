"""
JARVIS AI — Filtered Notification Synchronizer
==============================================
Synchronizes notifications between JARVIS and authorized companion devices.
Enforces category filtering:
- Approved Whitelist: urgent, reminder, alert, system_event
- Blocked Blacklist: otp, banking, credentials, unfiltered_app_push, auth, password

Provides offline queuing: notifications intended for temporarily offline devices
are persisted and delivered immediately upon reconnect or heartbeat.
"""

import uuid
import re
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from core.logger import log


DEFAULT_APPROVED_CATEGORIES = {
    "urgent",
    "reminder",
    "alert",
    "system_event",
    "status",
    "device_status",
}

DEFAULT_BLOCKED_CATEGORIES = {
    "otp",
    "banking",
    "credentials",
    "unfiltered_app_push",
    "auth",
    "password",
    "secret",
}


class NotificationItem(BaseModel):
    notification_id: str = Field(default_factory=lambda: f"notif-{uuid.uuid4().hex[:8]}")
    source_device: str
    target_device: str
    title: str
    message: str
    category: str
    priority: str = "normal"  # "low", "normal", "high", "urgent"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    delivered: bool = False
    delivered_at: Optional[datetime] = None


class NotificationSyncEngine:
    """Manages secure notification filtering, category controls, and offline queuing."""

    def __init__(self):
        self._allowed_categories: Set[str] = set(DEFAULT_APPROVED_CATEGORIES)
        self._blocked_categories: Set[str] = set(DEFAULT_BLOCKED_CATEGORIES)
        # target_device_id -> list of NotificationItem
        self._offline_queues: Dict[str, List[NotificationItem]] = {}
        # Chronological notification history
        self._history: List[NotificationItem] = []

    def is_category_permitted(self, category: str) -> bool:
        """
        Verify if notification category is allowed under the security policy.
        """
        clean = category.lower().strip()
        if clean in self._blocked_categories:
            return False
        # Keyword checks for sensitive tokens in category
        for blocked in self._blocked_categories:
            if blocked in clean:
                return False
        return clean in self._allowed_categories

    def configure_category(self, category: str, allowed: bool):
        """Allow user to toggle category permissions."""
        clean = category.lower().strip()
        if allowed:
            self._allowed_categories.add(clean)
            self._blocked_categories.discard(clean)
            log.info(f"Notification category [{clean}] enabled.")
        else:
            self._allowed_categories.discard(clean)
            self._blocked_categories.add(clean)
            log.info(f"Notification category [{clean}] blocked.")

    def list_allowed_categories(self) -> List[str]:
        """List currently permitted categories."""
        return sorted(list(self._allowed_categories))

    def send_notification(
        self,
        source_device: str,
        target_device: str,
        title: str,
        message: str,
        category: str = "system_event",
        priority: str = "normal",
        is_target_online: bool = True,
    ) -> NotificationItem:
        """
        Filter and dispatch notification.
        Raises PermissionError if category violates security policy.
        """
        clean_cat = category.lower().strip()
        if not self.is_category_permitted(clean_cat):
            log.warning(
                f"Notification blocked by category filter: category='{category}' "
                f"from device '{source_device}' to '{target_device}'."
            )
            raise PermissionError(
                f"Notification category '{category}' is blocked by security policy (sensitive or unapproved)."
            )

        # Inspect title and body for accidental credential leaks
        content_lower = f"{title} {message}".lower()
        if re.search(r"\b(otp|one[-\s]time password|verification code is \d+)\b", content_lower):
            raise PermissionError("Notification content rejected: sensitive OTP/credentials detected.")

        notif = NotificationItem(
            source_device=source_device,
            target_device=target_device,
            title=title,
            message=message,
            category=clean_cat,
            priority=priority,
            delivered=is_target_online,
            delivered_at=datetime.now(timezone.utc) if is_target_online else None,
        )

        self._history.append(notif)

        if not is_target_online:
            if target_device not in self._offline_queues:
                self._offline_queues[target_device] = []
            self._offline_queues[target_device].append(notif)
            log.info(f"Target device [{target_device}] offline; notification queued [{notif.notification_id}].")
        else:
            log.info(f"Notification dispatched to [{target_device}]: '{title}' ({clean_cat})")

        return notif

    def get_queued_notifications(
        self,
        device_id: str,
        mark_delivered: bool = True,
    ) -> List[NotificationItem]:
        """Retrieve queued notifications for a device upon reconnect."""
        queue = self._offline_queues.get(device_id, [])
        if not queue:
            return []

        now = datetime.now(timezone.utc)
        results = list(queue)

        if mark_delivered:
            for item in results:
                item.delivered = True
                item.delivered_at = now
            self._offline_queues[device_id] = []
            log.info(f"Delivered {len(results)} queued notifications to device [{device_id}].")

        return results

    def get_notification_history(self, limit: int = 50) -> List[NotificationItem]:
        """Fetch notification history."""
        return self._history[-limit:]


notification_sync_engine = NotificationSyncEngine()
