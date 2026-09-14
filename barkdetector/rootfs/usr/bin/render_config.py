#!/usr/bin/env python3
"""Translate Home Assistant add-on options into the detector's config file.

The add-on UI exposes a flat, friendly set of options; the detector expects the
nested YAML documented in ``config/example-config.yaml``. Secrets are
deliberately not written here -- MQTT credentials reach the app through
environment variables set by the service ``run`` script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# Captures land in /media so they show up in Home Assistant's Media browser and
# can be played back in the UI. /share is not browsable there, and /data would
# be private to this add-on.
CAPTURE_DIR = "/media/barkdetector"

# Built-in watched classes, keyed by the slug used as MQTT event name / HA
# sensor state. Substrings are matched case-insensitively against YAMNet's
# 521 display names. Add more groups here as you discover useful labels
# (see YAMNetClassifier.top_label for tuning new ones against real audio).
BUILTIN_WATCHED_CLASSES: dict[str, list[str]] = {
    "dog_bark": ["dog", "bark", "yip", "bow-wow", "howl"],
    "glass_break": ["glass", "shatter"],
    "alarm": ["alarm", "siren", "smoke detector", "fire alarm"],
    "scream": ["scream", "shout", "yell", "crying, sobbing"],
    "knock_doorbell": ["knock", "doorbell"],
}


def resolve_mic_device(raw: object) -> object:
    """Accept either a PortAudio device index or a substring of its name.

    The add-on option is a string because the HA options UI has no "int or
    text" type; an all-digit value is treated as an index, anything else is
    passed through for name matching, and empty means "system default".
    """
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    return value


def build_watched_classes(options: dict) -> list[dict]:
    """Turn the flat add-on options into a list of watched-class dicts.

    Two knobs, both optional and additive:

    - ``watched_events``: comma-separated slugs from BUILTIN_WATCHED_CLASSES
      to enable, e.g. "dog_bark,glass_break". All of them share the single
      ``conf_threshold`` option -- per-class thresholds are a config-file-only
      feature (edit the rendered YAML directly, or extend this function).
    - ``custom_watched_classes``: a JSON array for anything the built-ins
      don't cover, e.g.
      '[{"name": "car_horn", "label_substrings": ["car horn", "honk"],
      "conf_threshold": 0.5}]'. Entries here are appended as-is, each with
      its own threshold.
    """
    threshold = float(options.get("conf_threshold", 0.2))
    requested = str(options.get("watched_events", "dog_bark")).strip()
    slugs = [s.strip() for s in requested.split(",") if s.strip()]

    watched: list[dict] = []
    for slug in slugs:
        substrings = BUILTIN_WATCHED_CLASSES.get(slug)
        if substrings is None:
            print(
                f"Unknown watched_events entry '{slug}' (not in BUILTIN_WATCHED_CLASSES); skipping",
                file=sys.stderr,
            )
            continue
        watched.append(
            {"name": slug, "label_substrings": substrings, "conf_threshold": threshold}
        )

    custom_raw = str(options.get("custom_watched_classes", "")).strip()
    if custom_raw:
        try:
            custom_entries = json.loads(custom_raw)
            for entry in custom_entries:
                watched.append(
                    {
                        "name": str(entry["name"]),
                        "label_substrings": list(entry.get("label_substrings", [])),
                        "conf_threshold": float(entry.get("conf_threshold", threshold)),
                    }
                )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"Ignoring invalid custom_watched_classes ({exc}): {custom_raw}", file=sys.stderr)

    return watched


def build_config(options: dict) -> dict:
    mode = str(options.get("detection_mode", "yamnet")).lower()

    return {
        "device_id": options.get("device_id", "barkdetector"),
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "window_seconds": 0.975,
            "hop_seconds": 0.5,
            "mic_device_index": resolve_mic_device(options.get("mic_device")),
        },
        "detection": {
            "mode": mode,
            "yamnet": {
                "model_url": "https://storage.googleapis.com/audioset/yamnet/yamnet.tflite",
                "classes_url": (
                    "https://raw.githubusercontent.com/tensorflow/models/master/"
                    "research/audioset/yamnet/yamnet_class_map.csv"
                ),
                "watched_classes": build_watched_classes(options),
            },
            "normalize": {
                "enabled": bool(options.get("normalize_windows", False)),
                "target_peak": 0.5,
                "noise_floor": float(options.get("normalize_noise_floor", 0.005)),
                "max_gain": float(options.get("normalize_max_gain", 30)),
            },
            "heuristic": {
                "rms_threshold": 0.015,
                "band_low_hz": 400,
                "band_high_hz": 3000,
                "band_energy_min": 5.0e-6,
            },
        },
        "smoothing": {
            "window_count": int(options.get("windows_total", 5)),
            "positives_required": int(options.get("windows_required", 3)),
            "cooldown_seconds": int(options.get("cooldown_seconds", 10)),
        },
        "capture": {
            "enabled": bool(options.get("capture_enabled", False)),
            "ring_seconds": 20,
            "pre_seconds": 5,
            "post_seconds": 5,
            "normalize_peak": float(options.get("capture_normalize_peak", 0.7)),
            "out_dir": CAPTURE_DIR,
            "max_age_hours": float(options.get("capture_retention_hours", 24)),
            "max_total_mb": float(options.get("capture_max_mb", 512)),
        },
        # Placeholders only. The real values come from BARKDETECTOR_MQTT_*.
        "mqtt": {
            "host": "",
            "port": 1883,
            "topic": options.get("mqtt_topic", "home/sensors/dog_bark"),
            "username": "",
            "password": "",
            "client_id": "",
        },
        "dailybot": {"workflow_url": ""},
        "logging": {
            "level": str(options.get("log_level", "info")).upper(),
            # s6 captures stdout/stderr into the add-on log; a second copy on
            # disk would grow unbounded inside the container.
            "file_path": "/data/barkdetector.log",
        },
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: render_config.py <options.json> <output.yaml>", file=sys.stderr)
        return 2

    options_path, output_path = Path(sys.argv[1]), Path(sys.argv[2])

    try:
        options = json.loads(options_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Unable to read add-on options from {options_path}: {exc}", file=sys.stderr)
        return 1

    config = build_config(options)

    if config["capture"]["enabled"]:
        Path(CAPTURE_DIR).mkdir(parents=True, exist_ok=True)

    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Rendered detector configuration to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
