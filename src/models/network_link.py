"""Network link model.

A link is a bidirectional logical channel connecting a sender node to a
receiver node.  It exposes:

* A configurable bandwidth (bytes/s).
* Propagation delay (s) and protocol overhead (fraction of payload).
* Contention multiplier so that the same physical medium can be shared by
  several concurrent transfers, which is what the scheduler sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NetworkLink:
    """A logical network endpoint between two compute nodes."""

    sender_node_id: str
    receiver_node_id: str
    bandwidth_bytes_per_s: float = 1.0e7  # ~80 Mbps raw
    propagation_delay_s: float = 0.001
    protocol_overhead_fraction: float = 0.05
    contention: float = 1.0  # >1 slows everyone down.
    effective_bandwidth_bytes_per_s: float = field(init=False)

    def __post_init__(self):
        self.bandwidth_bytes_per_s = max(
            self.bandwidth_bytes_per_s, 1.0
        )
        self.protocol_overhead_fraction = min(
            0.99, max(0.0, self.protocol_overhead_fraction)
        )
        self.contention = max(0.0, self.contention)
        self.effective_bandwidth_bytes_per_s = (
            self.bandwidth_bytes_per_s / max(1.0, self.contention)
        )

    @property
    def transfer_time_s(self) -> float:
        """Return a representative transfer time for documentation."""
        return 0.0

    def transfer_duration(self, data_size_bytes: int) -> float:
        """Elapsed wall time for a transfer of *data_size_bytes* over this link."""
        if data_size_bytes <= 0:
            return 0.0
        usable = self.effective_bandwidth_bytes_per_s * (
            1.0 - self.protocol_overhead_fraction
        )
        duration = data_size_bytes / max(usable, 1e-12)
        return duration + self.propagation_delay_s

    def calc_transfer_energy(self, data_size_bytes: int, transmit_power_w: float, receive_power_w: float = 0.0) -> float:
        """Energy in Joules for a transfer over this link."""
        duration = self.transfer_duration(data_size_bytes)
        transmit = transmit_power_w * duration
        receive = receive_power_w * duration
        return transmit + receive

    def to_dict(self) -> dict:
        return {
            "sender_node_id": self.sender_node_id,
            "receiver_node_id": self.receiver_node_id,
            "bandwidth_bytes_per_s": self.bandwidth_bytes_per_s,
            "propagation_delay_s": self.propagation_delay_s,
            "protocol_overhead_fraction": self.protocol_overhead_fraction,
            "contention": self.contention,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NetworkLink":
        return cls(**data)
