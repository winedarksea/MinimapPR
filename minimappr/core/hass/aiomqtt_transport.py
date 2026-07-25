"""``MqttTransport`` backed by ``aiomqtt``.

``aiomqtt`` is an optional extra (``pip install -e '.[hass]'``), so the import is
guarded exactly as ``core/effectors/onvif_ptz.py`` guards ``onvif``: absent means
the bridge reports ``transport_available: false`` and publishes nothing, rather
than the whole app failing to import.

``aiomqtt.Client`` is designed as an async context manager. The transport keeps
the context object alive across ``connect``/``disconnect`` because the bridge's
lifecycle is not a lexical block — it connects once and publishes for hours.
"""

from __future__ import annotations

import importlib.util
import logging
from functools import lru_cache

from minimappr.core.hass.transport import (
    MqttPublish,
    MqttTransportConfig,
    MqttTransportError,
)

logger = logging.getLogger(__name__)

try:
    import aiomqtt  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional runtime dependency
    aiomqtt = None  # type: ignore[assignment]


@lru_cache(maxsize=1)
def aiomqtt_available() -> bool:
    """True when the optional MQTT client can be imported.

    Uses ``find_spec`` rather than the module object so a probe from a config or
    status path costs nothing when the package is absent (same approach as
    ``classifiers/availability.py``). Cached because an installed-or-not answer
    cannot change within a process, and the status endpoint polls it.
    """
    if aiomqtt is not None:
        return True
    try:
        return importlib.util.find_spec("aiomqtt") is not None
    except (ImportError, ValueError):
        return False


class AiomqttTransport:
    def __init__(self, config: MqttTransportConfig) -> None:
        self._config = config
        self._client = None

    @property
    def name(self) -> str:
        return "aiomqtt"

    async def connect(self) -> None:
        if aiomqtt is None:
            raise MqttTransportError("aiomqtt is not installed")
        will = None
        if self._config.will is not None:
            will = aiomqtt.Will(
                topic=self._config.will.topic,
                payload=self._config.will.payload.encode("utf-8"),
                qos=self._config.will.qos,
                retain=self._config.will.retain,
            )
        tls_params = None
        if self._config.tls_enabled:
            tls_params = aiomqtt.TLSParameters()
        client = aiomqtt.Client(
            hostname=self._config.host,
            port=self._config.port,
            username=self._config.username or None,
            password=self._config.password or None,
            identifier=self._config.client_id or None,
            keepalive=self._config.keepalive_seconds,
            will=will,
            tls_params=tls_params,
            tls_insecure=self._config.tls_insecure if self._config.tls_enabled else None,
        )
        try:
            await client.__aenter__()
        except Exception as exc:
            raise MqttTransportError(f"MQTT connect to {self._config.host}:{self._config.port} failed: {exc}") from exc
        self._client = client

    async def publish(self, message: MqttPublish) -> None:
        client = self._client
        if client is None:
            raise MqttTransportError("publish attempted before connect")
        try:
            await client.publish(
                message.topic,
                payload=message.payload.encode("utf-8"),
                qos=message.qos,
                retain=message.retain,
            )
        except Exception as exc:
            raise MqttTransportError(f"MQTT publish to {message.topic} failed: {exc}") from exc

    async def disconnect(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        try:
            await client.__aexit__(None, None, None)
        except Exception as exc:
            # Nothing useful to do about a failed teardown; the session is gone
            # either way and the caller is already on a shutdown path.
            logger.debug("aiomqtt disconnect raised: %s", exc)
