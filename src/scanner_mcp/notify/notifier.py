"""OS notifications and alert persistence (alerts are primary; agent reads via resource)."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def notify_desktop(title: str, message: str) -> None:
    try:
        from plyer import notification  # type: ignore[import-not-found]

        notification.notify(
            title=title[:120],
            message=message[:500],
            app_name="scanner-mcp",
            timeout=8,
        )
    except Exception as e:  # noqa: BLE001
        log.debug("plyer notification failed: %s", e)
