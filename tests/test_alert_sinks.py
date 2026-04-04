"""Tests for alert sinks — LogAlertSink, MqttAlertSink, and edge runtime integration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.config.models import EdgeConfig, MqttConfig, PlatformConfig
from src.transport.alert_sinks import AlertSinkBase, LogAlertSink, MqttAlertSink


# ---------------------------------------------------------------------------
# Sample alert fixture
# ---------------------------------------------------------------------------


def _sample_alert(**overrides) -> dict:
    defaults = {
        "timestamp": "2026-04-04T12:00:00+00:00",
        "channel_id": "ch_01",
        "score": 0.85,
        "fault": "overload_spike",
        "confidence": 0.9,
        "causes": ["Short-circuit protection fired"],
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# LogAlertSink
# ---------------------------------------------------------------------------


class TestLogAlertSink:
    def test_publish_logs_warning(self, caplog):
        sink = LogAlertSink()
        alert = _sample_alert()
        with caplog.at_level("WARNING"):
            sink.publish(alert)
        assert "overload_spike" in caplog.text
        assert "ch_01" in caplog.text

    def test_publish_batch(self, caplog):
        sink = LogAlertSink()
        alerts = [_sample_alert(channel_id=f"ch_{i:02d}") for i in range(3)]
        with caplog.at_level("WARNING"):
            sink.publish_batch(alerts)
        assert "ch_00" in caplog.text
        assert "ch_02" in caplog.text

    def test_close_is_noop(self):
        sink = LogAlertSink()
        sink.close()  # should not raise


# ---------------------------------------------------------------------------
# MqttAlertSink (mocked — no real broker needed)
# ---------------------------------------------------------------------------


class TestMqttAlertSink:
    def test_publish_calls_mqtt_client(self):
        """MqttAlertSink.publish should call client.publish with correct topic."""
        mock_client_class = MagicMock()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.publish.return_value = MagicMock(rc=0)

        mock_mqtt_module = MagicMock()
        mock_mqtt_module.Client = mock_client_class
        mock_mqtt_module.CallbackAPIVersion.VERSION2 = 2

        with patch.dict(
            "sys.modules",
            {"paho": MagicMock(), "paho.mqtt": MagicMock(), "paho.mqtt.client": mock_mqtt_module},
        ):
            sink = MqttAlertSink(broker_host="testhost", broker_port=1883, topic_prefix="vip/test")
            # Simulate connected state
            sink._connected = True
            sink._client = mock_client

            alert = _sample_alert()
            sink.publish(alert)

            mock_client.publish.assert_called_once()
            call_args = mock_client.publish.call_args
            assert call_args[0][0] == "vip/test/ch_01"  # topic
            assert "overload_spike" in call_args[0][1]  # payload contains fault

    def test_publish_when_disconnected_is_noop(self):
        """When not connected, publish should not raise."""
        sink = MqttAlertSink.__new__(MqttAlertSink)
        sink._client = None
        sink._connected = False
        sink.broker_host = "localhost"
        sink.broker_port = 1883
        sink.topic_prefix = "vip/alerts"
        sink.qos = 1
        # Should not raise
        sink.publish(_sample_alert())

    def test_close_disconnects(self):
        mock_client = MagicMock()
        sink = MqttAlertSink.__new__(MqttAlertSink)
        sink._client = mock_client
        sink._connected = True
        sink.broker_host = "localhost"
        sink.broker_port = 1883
        sink.close()
        mock_client.loop_stop.assert_called_once()
        mock_client.disconnect.assert_called_once()
        assert sink._client is None


# ---------------------------------------------------------------------------
# MqttConfig
# ---------------------------------------------------------------------------


class TestMqttConfig:
    def test_defaults(self):
        cfg = MqttConfig()
        assert cfg.enabled is False
        assert cfg.broker_host == "localhost"
        assert cfg.broker_port == 1883
        assert cfg.topic_prefix == "vip/alerts"
        assert cfg.qos == 1
        assert cfg.tls is False

    def test_platform_config_has_mqtt(self):
        cfg = PlatformConfig()
        assert hasattr(cfg, "mqtt")
        assert isinstance(cfg.mqtt, MqttConfig)
        assert cfg.mqtt.enabled is False


# ---------------------------------------------------------------------------
# Edge runtime with alert sinks
# ---------------------------------------------------------------------------


class TestEdgeRuntimeSinks:
    def test_alerts_published_to_sinks(self):
        """EdgeRuntime should call publish() on all alert sinks for each alert."""
        from src.edge.runtime import EdgeRuntime
        from src.inference.pipeline import InferencePipeline
        from src.transport.mock_can import DataFrameTransport

        rng = np.random.default_rng(42)
        n = 100
        t0 = datetime.now(tz=timezone.utc)
        # Create data with high spikes to trigger anomalies
        df = pd.DataFrame(
            {
                "timestamp": [t0 + timedelta(milliseconds=i * 100) for i in range(n)],
                "channel_id": "ch_01",
                "current_a": np.concatenate([rng.normal(5.0, 0.1, 50), rng.normal(25.0, 2.0, 50)]),
                "voltage_v": 13.5,
                "temperature_c": 40.0,
                "state_on_off": True,
                "trip_flag": [False] * 50 + [True] * 50,
                "overload_flag": [False] * 50 + [True] * 50,
                "reset_counter": 0,
                "pwm_duty_pct": 100.0,
            }
        )

        mock_sink = MagicMock(spec=AlertSinkBase)
        transport = DataFrameTransport(df)
        pipeline = InferencePipeline()
        edge_cfg = EdgeConfig(batch_size=25, alert_cooldown_s=0)
        runtime = EdgeRuntime(transport, pipeline, edge_cfg, alert_sinks=[mock_sink])

        alerts = runtime.run()

        # If any alerts were emitted, sink.publish should have been called
        if len(alerts) > 0:
            assert mock_sink.publish.call_count == len(alerts)
            # Verify alert dict structure
            first_call_alert = mock_sink.publish.call_args_list[0][0][0]
            assert "channel_id" in first_call_alert
            assert "fault" in first_call_alert
            assert "score" in first_call_alert

        # Sink should be closed on exit
        mock_sink.close.assert_called_once()

    def test_sink_failure_does_not_crash_runtime(self):
        """A failing sink should not bring down the edge runtime."""
        from src.edge.runtime import EdgeRuntime
        from src.inference.pipeline import InferencePipeline
        from src.transport.mock_can import DataFrameTransport

        rng = np.random.default_rng(42)
        n = 100
        t0 = datetime.now(tz=timezone.utc)
        df = pd.DataFrame(
            {
                "timestamp": [t0 + timedelta(milliseconds=i * 100) for i in range(n)],
                "channel_id": "ch_01",
                "current_a": np.concatenate([rng.normal(5.0, 0.1, 50), rng.normal(25.0, 2.0, 50)]),
                "voltage_v": 13.5,
                "temperature_c": 40.0,
                "state_on_off": True,
                "trip_flag": [False] * 50 + [True] * 50,
                "overload_flag": [False] * 50 + [True] * 50,
                "reset_counter": 0,
                "pwm_duty_pct": 100.0,
            }
        )

        # Sink that always raises
        failing_sink = MagicMock(spec=AlertSinkBase)
        failing_sink.publish.side_effect = ConnectionError("broker down")

        transport = DataFrameTransport(df)
        pipeline = InferencePipeline()
        edge_cfg = EdgeConfig(batch_size=25, alert_cooldown_s=0)
        runtime = EdgeRuntime(transport, pipeline, edge_cfg, alert_sinks=[failing_sink])

        # Should complete without raising
        alerts = runtime.run()
        assert isinstance(alerts, list)
