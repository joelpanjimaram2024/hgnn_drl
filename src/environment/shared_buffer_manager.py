"""Shared-buffer (hyperedge) manager.

Tracks buffer ownership and reuse so that transfer costs are
count-consistently: when several tasks reuse a transferred buffer, the cost
is charged once (or proportionally to the reuse pattern), avoiding
double counting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..utils.config import Config
from ..utils.logging import get_logger


@dataclass
class BufferState:
    """Internal state of one shared buffer."""

    buffer_id: str
    owner_node_id: Optional[str] = None          # which node currently holds it
    transferred_amounts: Dict[str, int] = field(default_factory=dict)  # node -> cumulative bytes
    last_access_time: Optional[float] = None
    transfer_count: int = 0


class SharedBufferManager:
    """Manages shared-data buffers and their reuse across tasks."""

    def __init__(self, *, config: Config):
        self.config = config
        self.logger = get_logger("hgnn_drl.shared_buffer")
        self.buffers: Dict[str, BufferState] = {}
        self.total_transfer_bytes: int = 0
        self.total_transfers: int = 0

    def register_buffer(self, buffer_id: str, owner_node_id: Optional[str] = None) -> None:
        if buffer_id not in self.buffers:
            self.buffers[buffer_id] = BufferState(buffer_id=buffer_id, owner_node_id=owner_node_id)
        elif self.buffers[buffer_id].owner_node_id is None:
            self.buffers[buffer_id].owner_node_id = owner_node_id

    def note_access(
        self,
        *,
        buffer_id: str,
        task_id: str,
        node_id: Optional[str],
        bytes_accessed: int,
    ) -> int:
        """Record a task's access to a buffer and return incremental cost.

        The cost is charged only for data not previously supplied by this
        node, so multi-task reuse does not inflate the transfer cost.
        """
        if buffer_id not in self.buffers:
            self.buffers[buffer_id] = BufferState(buffer_id=buffer_id, owner_node_id=node_id)
        state = self.buffers[buffer_id]

        state.last_access_time = None  # set by caller for time
        incremental = bytes_accessed
        if node_id is not None and state.owner_node_id == node_id:
            # Already present locally; zero incremental transfer.
            incremental = 0
        else:
            state.owner_node_id = node_id
            state.transferred_amounts[node_id] = state.transferred_amounts.get(node_id, 0) + bytes_accessed
            state.transfer_count += 1
            self.total_transfer_bytes += int(bytes_accessed)
            self.total_transfers += 1
        return incremental

    def buffer_sizes(self) -> Dict[str, int]:
        """Return a deterministic mapping buffer_id -> size hint."""
        return {bid: 0 for bid in self.buffers}

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_transfer_bytes": self.total_transfer_bytes,
            "total_transfers": self.total_transfers,
            "buffer_count": len(self.buffers),
        }
