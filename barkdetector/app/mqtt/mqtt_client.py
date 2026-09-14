"""MQTT publishing helper."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from loguru import logger
from paho.mqtt import client as mqtt


@dataclass
class MQTTConfig:
    host: str
    port: int
    topic: str
    username: str | None = None
    password: str | None = None


class MQTTPublisher:
    """Publish events to MQTT with automatic reconnection."""

    def __init__(self, config: MQTTConfig, client_id: Optional[str] = None) -> None:
        self.config = config

        # Generate unique client ID if none provided to avoid conflicts
        if not client_id or client_id.strip() == "":
            client_id = f"barkdetector-{uuid.uuid4().hex[:8]}"
            logger.warning(
                "No client_id configured; auto-generated unique ID: {}", client_id
            )

        self.client = mqtt.Client(client_id=client_id, clean_session=True)
        if config.username:
            self.client.username_pw_set(config.username, config.password or "")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self._connected = threading.Event()
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

    def start(self) -> None:
        """Connect to the broker and start the network loop."""
        try:
            self.client.connect(self.config.host, int(self.config.port), keepalive=120)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("Initial MQTT connection failed: {}", exc)
        self.client.loop_start()

    def stop(self) -> None:
        """Stop the loop and disconnect."""
        self.client.loop_stop()
        try:
            self.client.disconnect()
        except Exception:
            pass

    def publish(self, payload: dict, qos: int = 1, retain: bool = False) -> None:
        """Publish a JSON payload to the configured topic."""
        data = json.dumps(payload)
        if not self._connected.wait(timeout=2.0):
            logger.warning("MQTT client not connected; attempting publish anyway")
        result = self.client.publish(self.config.topic, data, qos=qos, retain=retain)
        if result.rc not in (mqtt.MQTT_ERR_SUCCESS, mqtt.MQTT_ERR_NO_CONN):
            logger.error("MQTT publish failed with code {}", result.rc)
        else:
            logger.debug("Published to MQTT topic {} with QoS {}", self.config.topic, qos)

    def publish_discovery(self, device_id: str, device_name: str = "Sound Detector") -> None:
        """Announce a generic sound-classification sensor via HA MQTT discovery.

        Unlike the original single-purpose "Bark" binary_sensor, this is a
        plain ``sensor`` whose *state* is the name of whichever watched class
        last matched (e.g. ``dog_bark``, ``glass_break``, ``siren``). That
        makes it trivial to trigger automations directly on state, e.g.::

            trigger:
              - platform: state
                entity_id: sensor.<device_id>_sound
                to: "glass_break"

        The sensor has no "off"/idle value -- it simply holds the last
        detected class until the next one arrives. A companion
        ``last_triggered`` attribute (epoch seconds) is published alongside
        it so automations that need to react to *repeated* occurrences of
        the same class in a row (state doesn't change, so a plain state
        trigger won't refire) can instead watch that attribute.
        """
        object_id = "".join(c if c.isalnum() else "_" for c in device_id).strip("_")
        config_topic = f"homeassistant/sensor/{object_id}/sound/config"
        payload = {
            "name": "Sound",
            "unique_id": f"{object_id}_sound",
            "state_topic": self.config.topic,
            "value_template": "{{ value_json.event }}",
            "json_attributes_topic": self.config.topic,
            "json_attributes_template": (
                "{{ {'score': value_json.score, 'last_triggered': value_json.ts, "
                "'detector': value_json.detector} | tojson }}"
            ),
            "icon": "mdi:ear-hearing",
            # Without this, HA silently drops a new MQTT message whose value
            # equals the current state -- no history entry, no state_changed
            # event, no automation trigger. Two glass_break events in a row
            # (or any repeat of the last detected class) would otherwise
            # vanish even though MQTT delivered them successfully.
            "force_update": True,
            "device": {
                "identifiers": [object_id],
                "name": device_name,
                "manufacturer": "barkdetector",
                "model": "YAMNet microphone sensor",
            },
        }

        data = json.dumps(payload)
        if not self._connected.wait(timeout=5.0):
            logger.warning("Not connected; discovery config may not reach the broker")
        result = self.client.publish(config_topic, data, qos=1, retain=True)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.info("Published MQTT discovery config to {}", config_topic)
        else:
            logger.error("Failed to publish discovery config (code {})", result.rc)

    # Callbacks -------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc):  # type: ignore[override]
        if rc == 0:
            logger.info("Connected to MQTT broker at {}:{}", self.config.host, self.config.port)
            self._connected.set()
        else:
            logger.error("MQTT connection refused (code {})", rc)

    def _on_disconnect(self, client, userdata, rc):  # type: ignore[override]
        if rc != 0:
            logger.warning("Unexpected MQTT disconnection (code {}), retrying", rc)
            # give some time before declaring offline to avoid thrashing
            time.sleep(1.0)
        self._connected.clear()
