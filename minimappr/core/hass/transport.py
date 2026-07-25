"""MQTT transport abstraction: the Protocol every client implementation satisfies.

Mirrors ``core/effectors/base.py`` — a narrow Protocol plus frozen value
dataclasses, so the bridge never imports a concrete MQTT client and tests can
substitute an in-memory recorder at the ``bridge._build_transport`` seam.

``MqttPublish`` carries the two policy flags the bridge's flush logic keys off:

* ``retain`` — set by ``state_mapper``, never by the bridge, so retain policy is
  golden-testable alongside the payload it applies to.
* ``coalescable`` — True for stateful topics (last-write-wins within a flush,
  suppressed when unchanged); False for impulses (events, alerts, rule actions)
  which must fire even when byte-identical to the previous publish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class MqttPublish:
    topic: str
    payload: str
    retain: bool = False
    qos: int = 0
    coalescable: bool = True
    """False for impulses (event entities, alerts, rule actions): dedupe/coalesce must not swallow them."""


@dataclass(frozen=True, slots=True)
class MqttWill:
    """Last Will and Testament — the broker publishes this if we vanish."""
    topic: str
    payload: str
    retain: bool = True
    qos: int = 0


@dataclass(frozen=True, slots=True)
class MqttTransportConfig:
    host: str
    port: int = 1883
    username: str = ""
    password: str = ""
    client_id: str = "minimappr"
    keepalive_seconds: int = 60
    tls_enabled: bool = False
    tls_insecure: bool = False
    will: MqttWill | None = None


class MqttTransportError(RuntimeError):
    pass


@runtime_checkable
class MqttTransport(Protocol):
    @property
    def name(self) -> str:
        ...

    async def connect(self) -> None:
        """Establish a session. Raises ``MqttTransportError`` on failure."""
        ...

    async def publish(self, message: MqttPublish) -> None:
        """Publish one message. Raises ``MqttTransportError`` on failure."""
        ...

    async def disconnect(self) -> None:
        """Tear the session down. Must be safe to call when never connected."""
        ...
