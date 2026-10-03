"""Mist-only baseline."""

from .base import BaselineScheduler


class MistOnlyScheduler(BaselineScheduler):
    """Attempt to execute tasks on the local Mist node."""

    layer = "Mist"
