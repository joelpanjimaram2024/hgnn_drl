"""Cloud-only baseline."""

from .base import BaselineScheduler


class CloudOnlyScheduler(BaselineScheduler):
    """Send every eligible task to the Cloud (subject to simulator constraints)."""

    layer = "Cloud"
