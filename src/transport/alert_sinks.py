"""Alert sinks — pluggable destinations for VIP alerts.

Provides a base interface and concrete implementations:
  - LogAlertSink      — writes alerts via structured logging (default)
  - MqttAlertSink     — publishes alerts as JSON to an MQTT broker
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from src.utils.logging import get_logger

log = get_logger(__name__)


class AlertSinkBase(ABC):
    """Interface for alert delivery backends."""

    @abstractmethod
    def publish(self, alert: dict) -> None:
        """Deliver a single alert."""
        ...

    def publish_batch(self, alerts: list[dict]) -> None:
        """Deliver multiple alerts. Default: iterate and publish each."""
        for a in alerts:
            self.publish(a)

    def close(self) -> None:
        """Release resources. Override in subclasses that hold connections."""
        pass


class LogAlertSink(AlertSinkBase):
    """Emits alerts via the structured logger (always available, zero config)."""

    def publish(self, alert: dict) -> None:
        log.warning(
            "ALERT: %s on %s (score=%.2f)",
            alert.get("fault", "unknown"),
            alert.get("channel_id", "unknown"),
            alert.get("score", 0),
        )


class MqttAlertSink(AlertSinkBase):
    """Publishes alerts as JSON messages to an MQTT broker.

    Requires the ``paho-mqtt`` package (``pip install paho-mqtt``).
    Falls back gracefully if the broker is unreachable — alerts are logged
    but not lost (they still go to the local alerts.json via StorageWriter).
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic_prefix: str = "vip/alerts",
        client_id: str = "",
        username: str = "",
        password: str = "",
        qos: int = 1,
        tls: bool = False,
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic_prefix = topic_prefix.rstrip("/")
        self.qos = qos
        self._connected = False
        self._client = None

        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            log.error("paho-mqtt not installed — run: pip install paho-mqtt")
            return

        cid = client_id or f"vip-edge-{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=cid,
        )

        if username:
            self._client.username_pw_set(username, password)
        if tls:
            self._client.tls_set()

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        try:
            self._client.connect(broker_host, broker_port, keepalive=60)
            self._client.loop_start()
        except Exception as exc:
            log.warning("MQTT connect failed (%s:%d): %s — alerts will be logged only",
                        broker_host, broker_port, exc)
            self._client = None

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        self._connected = True
        log.info("MQTT connected to %s:%d", self.broker_host, self.broker_port)

    def _on_disconnect(self, client, userdata, flags, rc, properties=None) -> None:
        self._connected = False
        if rc != 0:
            log.warning("MQTT disconnected unexpectedly (rc=%s)", rc)

    def publish(self, alert: dict) -> None:
        if self._client is None or not self._connected:
            log.debug("MQTT not connected — alert logged locally only")
            return

        channel_id = alert.get("channel_id", "unknown")
        topic = f"{self.topic_prefix}/{channel_id}"
        payload = json.dumps(alert, default=str)

        try:
            result = self._client.publish(topic, payload, qos=self.qos)
            if result.rc != 0:
                log.warning("MQTT publish failed (rc=%d) for %s", result.rc, topic)
        except Exception as exc:
            log.warning("MQTT publish error: %s", exc)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
            self._connected = False
            log.info("MQTT client closed")
